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
    # wait until checkpoints in all processes are loaded
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
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if args.rank == 0:
        torch.save(cpu_state, name)
    torch.distributed.barrier()


def train_one_epoch(dataloader, optimizer, llava_model, tokenizer, loss_fn, args):
    llava_model = llava_model.train()
    pbar = tqdm(dataloader)
    for sample in pbar:
        feature_dict = EasyDict(
            scene_feature=sample.scene_feature.to("cpu"),
            scene_insert_loc=sample.scene_insert_loc,
            scene_length=sample.scene_length,
        )
        input_ids = sample.input_ids.to("cpu")
        attention_mask = sample.attention_mask.to("cpu")
        labels = input_ids.clone()
        answer_indices = torch.where(labels == 22550)[1]

        print(answer_indices)

        for j, answer_idx in enumerate(answer_indices):
            labels[j, : answer_idx + 2] = -100

        labels[labels == tokenizer.pad_token_id] = -100

        # decode all the tokens with token_id != -100 back to text and print them out
        # print(tokenizer.decode(input_ids[0][input_ids[0] != tokenizer.pad_token_id]))
        # print(tokenizer.decode(labels[0][labels[0] != -100]))
        # print(labels[0])
        # print(tokenizer.decode(input_ids[1][input_ids[1] != tokenizer.pad_token_id]))
        # print(tokenizer.decode(labels[1][labels[1] != -100]))
        # print(labels[1])

        optimizer.zero_grad()

        outputs = llava_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            feature_dict=feature_dict,
            output_hidden_states=True,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.3f}")


def eval(dataloader, model, tokenizer):
    model.eval()
    total = 0
    correct = 0
    pbar = tqdm(dataloader)
    max_token_length = 0
    with torch.no_grad():
        for sample in pbar:
            input_ids = sample.input_ids
            answer_ind = torch.where(sample.input_ids == 22550)[1][0].item()
            answer_ids = input_ids[:, answer_ind + 2 : answer_ind + 6]
            # print(tokenizer.decode(answer_ids[0]))
            input_ids = input_ids[:, : answer_ind + 2]
            max_token_length = max(max_token_length, input_ids.shape[1])
            feature_dict = EasyDict(
                scene_feature=sample.scene_feature.to("cpu"),
                scene_insert_loc=sample.scene_insert_loc,
                scene_length=sample.scene_length,
            )
            input_ids = input_ids.to("cpu")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    feature_dict=feature_dict,
                    do_sample=False,
                    max_new_tokens=10,
                )
            outputs = (
                tokenizer.decode(output_ids[0, input_ids.shape[1] :])
                .replace("</s>", "")
                .strip()
            )
            gt = tokenizer.decode(answer_ids[0]).replace("</s>", "").strip()
            total += 1
            if gt.lower().strip() == outputs.lower().strip():
                correct += 1

            pbar.set_description(f"acc: {correct / total}")
    print(max_token_length)


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
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--folder", default="tmp", help="save folder")
    args = parser.parse_args()
    # args.local_rank, args.rank, args.world_size = world_info_from_env()
    # print(f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}")
    # device_id = init_distributed_device(args)

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

    # from dataset import (
    #     SCENE_TOKEN, SELECT_TOKEN
    # )
    # additional_special_tokens = [SCENE_TOKEN, SELECT_TOKEN]
    # tokenizer.add_tokens(additional_special_tokens, special_tokens=True)

    dataset = ExploreDataset("../exploregraph_data", tokenizer, 2048)
    train_index, test_index = dataset.split_index(test_ratio=0.25)
    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, test_index)
    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        collate_fn=dataset.collate_wrapper,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        collate_fn=dataset.collate_wrapper,
    )
    all_dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=dataset.collate_wrapper,
    )

    # freeze model
    model.requires_grad_(True)
    del model.model.vision_tower
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # wrap model and optimizer with DDP

    model = model.to("cpu")

    loss_fn = torch.nn.CrossEntropyLoss()
    # start training
    for epoch in range(args.num_epochs):
        print("Start training epoch %d" % epoch)
        # train_one_epoch(dataloader, optimizer, model, tokenizer, loss_fn, args)
        # save checkpoint
        # save_checkpoint(model, args.folder, epoch, args)
        print("evaluating")
        eval(val_dataloader, model, tokenizer)


if __name__ == "__main__":
    main()
