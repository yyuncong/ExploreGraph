""" Main training script """

import argparse
import copy
import glob
import os
import random
import functools
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from dataset import ExploreDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from easydict import EasyDict
from accelerate import load_checkpoint_and_dispatch

import numpy as np
import torch
from distributed import init_distributed_device, world_info_from_env
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import warnings

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %I:%M:%S",
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from tqdm import tqdm
import torch.nn.functional as F


def load_checkpoint(model, args, name="checkpoint.pt"):
    checkpoint = torch.load(name, map_location="cpu")
    torch.distributed.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint, True)
    del checkpoint
    torch.cuda.empty_cache()
    torch.distributed.barrier()


def save_checkpoint(model, folder, epoch, args, name="checkpoint.pt"):
    try:
        if not os.path.exists(folder):
            os.mkdir(folder)
    except:
        pass
    name = os.path.join(folder, "checkpoint_%d.pt" % epoch)
    # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    # with FSDP.state_dict_type(
    #    model, StateDictType.FULL_STATE_DICT, save_policy
    # ):
    cpu_state = model.state_dict()
    if args.rank == 0:
        torch.save(cpu_state, name)
    # torch.distributed.barrier()


def train_one_epoch(dataloader, optimizer, llava_model, tokenizer, loss_fn, args):
    llava_model = llava_model.train()
    pbar = tqdm(dataloader)
    for sample in pbar:
        feature_dict = EasyDict(
            scene_feature=sample.scene_feature.to("cuda"),
            scene_insert_loc=sample.scene_insert_loc,
            scene_length=sample.scene_length,
        )
        input_ids = sample.input_ids.to("cuda")
        attention_mask = sample.attention_mask.to("cuda")
        labels = input_ids.clone()
        answer_indices = torch.where(labels == 22550)[1]

        # input parts are not considered
        for j, answer_idx in enumerate(answer_indices):
            labels[j, : answer_idx + 2] = -100

        labels[labels == tokenizer.pad_token_id] = -100
        optimizer.zero_grad()

        # with torch.autocast(device_type="cuda"):
        # scene_feature = llava_model.model.mm_projector(sample.scene_feature.to("cuda"))
        # feature_dict.scene_feature_proj = scene_feature
        # can it automatically replace special token embeddings with scene feature?
        outputs = llava_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            feature_dict=feature_dict,
            output_hidden_states=True,
        )
        hidden_state = outputs["hidden_states"][-1][:, -1, :].unsqueeze(1)
        # project feature for attention computation
        scene_feature = llava_model.model.mm_projector(sample.scene_feature.to("cuda"))
        attention = torch.einsum("abf,acf-> abc", scene_feature, hidden_state).squeeze(
            -1
        )
        prediction = sample.prediction.to("cuda")

        # weights for loss computation
        weights = torch.zeros_like(attention)
        weights[prediction == 0] = 0.2
        weights[prediction == 1] = 1
        weights[prediction == -1] = 0

        for i in range(prediction.shape[0]):
            weights[i][sample["max_scene_length"][i] :] = 0

        attention = attention.reshape(-1).to("cuda")
        prediction = prediction.reshape(-1).to("cuda")
        weights = weights.reshape(-1).to("cuda")

        pos_weight = (torch.ones(attention.shape) * 5).to("cuda")

        loss2 = F.binary_cross_entropy_with_logits(
            attention, prediction, weight=weights, pos_weight=pos_weight
        )

        loss = outputs.loss
        # loss += loss2
        # loss.backward()
        loss = torch.tensor(0).to("cuda")
        loss2.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.3f} loss2: {loss2.item():.3f} ")


def eval(dataloader, model, tokenizer):
    model.eval()
    total = 0
    correct = 0
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for sample in pbar:
            input_ids = sample.input_ids
            attention_mask = sample.attention_mask
            labels = input_ids.clone()
            # seems we do not need things like answer?
            answer_indices = torch.where(labels == 22550)[1]
            for j, answer_idx in enumerate(answer_indices):
                labels[j, : answer_idx + 2] = -100
            labels[labels == tokenizer.pad_token_id] = -100

            answer_ind = torch.where(sample.input_ids == 22550)[1][0].item()
            answer_ids = input_ids[:, answer_ind + 2 :]
            # input_ids = input_ids[:, :answer_ind+2]
            # compute answer
            feature_dict = EasyDict(
                scene_feature=sample.scene_feature.to("cuda"),
                scene_insert_loc=sample.scene_insert_loc,
                scene_length=sample.scene_length,
            )
            input_ids = input_ids.to("cuda")
            with torch.inference_mode() and torch.autocast(device_type="cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    feature_dict=feature_dict,
                    output_hidden_states=True,
                )
                hidden_state = outputs["hidden_states"][-1][:, -1, :].unsqueeze(1)
                scene_feature = model.model.mm_projector(
                    sample.scene_feature.to("cuda")
                )
                attention = torch.einsum(
                    "abf,acf-> abc", scene_feature, hidden_state
                ).squeeze(-1)
                prediction = sample.prediction.to("cuda")

            # print(attention.shape)
            # print(prediction.shape)
            # calculate accuracy using attention and prediction
            output_ids = torch.argmax(attention, dim=-1)
            prediction_ids = torch.argmax(prediction, dim=-1)
            output_ids = output_ids.reshape(-1)
            prediction_ids = prediction_ids.reshape(-1)
            # print(output_ids.shape)
            # print(prediction_ids.shape)
            # print(output_ids)
            # print(prediction_ids)
            correct += torch.sum(prediction_ids == output_ids).item()
            total += prediction.shape[0]

            pbar.set_description(f"acc: {correct / total}")

    print("accuracy:", correct / total)

    # else:
    #     print("wrong:", gt, outputs)
    # if total % 10 == 0:
    #     print("accuracy:", correct / total)


def main():
    parser = argparse.ArgumentParser()
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--scene_path", 
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory", 
        help="scene path"
    )
    parser.add_argument(
        "--exploration_path",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/",
        help="exploration path"
    )
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--folder", default="tmp", help="save folder")
    args = parser.parse_args()
    # args.local_rank, args.rank, args.world_size = world_info_from_env()
    # print(f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}")
    # device_id = init_distributed_device(args)

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    print(
        f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}"
    )

    model_path = "liuhaotian/llava-v1.5-7b"
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=None, add_multisensory_token=True
    )
    # from dataset import (
    #     SCENE_TOKEN
    # )
    # additional_special_tokens = [SCENE_TOKEN]
    # tokenizer.add_tokens(additional_special_tokens, special_tokens=True)

    dataset = ExploreDataset(
        scene_path=args.scene_path,
        exploration_path=args.exploration_path,
        tokenizer=tokenizer, 
        max_length=2048,
    )
    train_index, test_index = dataset.split_index(test_ratio=0.25)
    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, test_index)
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=dataset.collate_wrapper,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        collate_fn=dataset.collate_wrapper,
    )

    # freeze model (only train the projector?)
    model.requires_grad_(False)
    model.model.mm_projector.requires_grad_(True)
    # model.requires_grad_(True)
    del model.model.vision_tower
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # wrap model and optimizer with DDP

    model = model.to("cuda")

    loss_fn = torch.nn.CrossEntropyLoss()
    # start training
    for epoch in range(args.num_epochs):
        print("Start training epoch %d" % epoch)
        # train_one_epoch(dataloader, optimizer, model, tokenizer, loss_fn, args)
        # save checkpoint
        # save_checkpoint(model, args.folder, epoch, args)
        print("evaluating")
        eval(dataloader, model, tokenizer)


if __name__ == "__main__":
    main()
