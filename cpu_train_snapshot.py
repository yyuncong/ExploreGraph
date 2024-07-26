""" Main training script """

import argparse
import copy
import glob
import os
import random
import functools
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from dataset_snapshot import ExploreDataset
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

logging_path = "log.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(logging_path, mode="w"),
        logging.StreamHandler(),
    ],
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from tqdm import tqdm
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        for j, answer_idx in enumerate(answer_indices):
            labels[j, : answer_idx + 2] = -100

        labels[labels == tokenizer.pad_token_id] = -100

        # Jiachen TODO: check the content of your new prompt by uncommenting the following line

        # print(tokenizer.decode(input_ids[0][input_ids[0] != tokenizer.pad_token_id]))
        # print()
        # print(tokenizer.decode(labels[0][labels[0] != -100]))
        logging.info(
            tokenizer.decode(input_ids[0][input_ids[0] != tokenizer.pad_token_id])
        )
        logging.info(tokenizer.decode(labels[0][labels[0] != -100]))

        # optimizer.zero_grad()

        # outputs = llava_model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels,
        #     feature_dict=feature_dict,
        #     output_hidden_states=True,
        # )
        # selection_loss = outputs.loss
        # combined_loss = selection_loss
        # Jiachen TODO: get the extra filter outputs with everything you added
        # and calculate the filter_loss and combine it with the total loss for training
        # Add the values of the two losses to the set_description line
        # None feature dict as a placeholder
        if args.prefiltering:
            filter_input_ids = sample.filter_input_ids.to("cpu")
            filter_attention_mask = sample.filter_attention_mask.to("cpu")
            filter_labels = filter_input_ids.clone()
            # choose the first answer as the separator
            filter_answer_indices = torch.where(filter_labels == 22550)[1]
            for j, answer_idx in enumerate(filter_answer_indices):
                filter_labels[j, : answer_idx + 2] = -100
            filter_labels[filter_labels == tokenizer.pad_token_id] = -100

            # test output
            # print(
            #     tokenizer.decode(
            #         filter_input_ids[0][filter_input_ids[0] != tokenizer.pad_token_id]
            #     )
            # )
            # print()
            # print(
            #     tokenizer.decode(
            #         filter_labels[0][filter_labels[0] != -100]
            #     )
            # )
            logging.info(
                tokenizer.decode(
                    filter_input_ids[0][filter_input_ids[0] != tokenizer.pad_token_id]
                )
            )
            logging.info(tokenizer.decode(filter_labels[0][filter_labels[0] != -100]))

            # filter_outputs = llava_model(
            #     input_ids=filter_input_ids,
            #     attention_mask=filter_attention_mask,
            #     labels=filter_labels,
            #     feature_dict=None,
            #     output_hidden_states=True,
            # )
            # filter_loss = filter_outputs.loss
            # combined_loss += filter_loss
        # combined_loss.backward()
        # optimizer.step()
        # if args.prefiltering:
        #     pbar.set_description(
        #         f"loss: {combined_loss.item():.3f}, selection_loss: {selection_loss.item():.3f}, filter_loss: {filter_loss.item():.3f}"
        #     )
        # else:
        #     pbar.set_description(f"loss: {combined_loss.item():.3f}")


def eval(dataloader, model, tokenizer, args):
    model.eval()
    total_combined_loss = 0
    total_selection_loss = 0
    total_filter_loss = 0
    total_sample = 0
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for sample in pbar:
            # calculate selection loss
            feature_dict = EasyDict(
                scene_feature=sample.scene_feature.to("cpu"),
                scene_insert_loc=sample.scene_insert_loc,
                scene_length=sample.scene_length,
            )
            input_ids = sample.input_ids.to("cpu")
            attention_mask = sample.attention_mask.to("cpu")
            labels = input_ids.clone()
            answer_indices = torch.where(labels == 22550)[1]

            for j, answer_idx in enumerate(answer_indices):
                labels[j, : answer_idx + 2] = -100

            labels[labels == tokenizer.pad_token_id] = -100
            with torch.autocast(device_type="cpu"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    feature_dict=feature_dict,
                    output_hidden_states=True,
                )
            selection_loss = outputs.loss
            combined_loss = selection_loss
            # calculate filter loss
            if args.prefiltering:
                filter_input_ids = sample.filter_input_ids.to("cpu")
                filter_attention_mask = sample.filter_attention_mask.to("cpu")
                filter_labels = filter_input_ids.clone()
                filter_answer_indices = torch.where(filter_labels == 22550)[1]
                for j, answer_idx in enumerate(filter_answer_indices):
                    filter_labels[j, : answer_idx + 2] = -100
                filter_labels[filter_labels == tokenizer.pad_token_id] = -100
                # test output
                """
                print(
                    tokenizer.decode(
                        filter_input_ids[0][filter_input_ids[0] != tokenizer.pad_token_id]
                    )
                )
                """
                with torch.autocase(device_type="cpu"):
                    filter_outputs = model(
                        input_ids=filter_input_ids,
                        attention_mask=filter_attention_mask,
                        labels=filter_labels,
                        feature_dict=None,
                        output_hidden_states=True,
                    )
                filter_loss = filter_outputs.loss
                combined_loss += filter_loss
            total_combined_loss += combined_loss.item()
            total_selection_loss += selection_loss.item()
            total_sample += input_ids.shape[0]
            if args.prefiltering:
                total_filter_loss += filter_loss.item()
                pbar.set_description(
                    f"loss: {total_combined_loss / total_sample:.3f}, selection_loss: {total_selection_loss / total_sample:.3f}, filter_loss: {total_filter_loss / total_sample:.3f}"
                )
            else:
                pbar.set_description(f"loss: {total_combined_loss / total_sample:.3f}")


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
    parser.add_argument(
        "--scene_path",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory",
        help="scene path",
    )
    parser.add_argument(
        "--exploration_path",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/",
        help="exploration path",
    )
    parser.add_argument(
        "--egocentric_views",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--action_memory",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--random_permute",
        action="store_true",
        help="if set true, randomly permute object/frontiers/pre-filtering classes",
        default=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    # Jiachen
    # Jiachen TODO: Add parameters for your feature
    # 1. Whether we are going to use the prefiltering
    # 2. How many object categories we are going to keep (5? 10? 20?)
    parser.add_argument("--prefiltering", action="store_true", default=False)
    parser.add_argument("--top_k_categories", type=int, default=5)
    parser.add_argument(
        "--add_positional_encodings", action="store_true", default=False
    )
    args = parser.parse_args()
    # set up random seed
    set_seed(args.seed)
    # args.local_rank, args.rank, args.world_size = world_info_from_env()
    # print(f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}")
    # device_id = init_distributed_device(args)

    model_path = "/gpfs/u/home/LMCG/LMCGhazh/scratch/external/yuncong/llava-v1.5-7b"
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

    # Jiachen TODO: pass your parameter in the dataset file
    dataset = ExploreDataset(
        scene_path=args.scene_path,
        exploration_path=args.exploration_path,
        egocentric_views=args.egocentric_views,
        action_memory=args.action_memory,
        prefiltering=args.prefiltering,
        top_k_categories=args.top_k_categories,
        random_permute=args.random_permute,
        add_positional_encodings=args.add_positional_encodings,
        tokenizer=tokenizer,
        max_length=2048,
    )
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
        train_dataset,
        batch_size=1,
        shuffle=True,
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

        # Jiachen TODO: update train_one_epoch for your feature
        dataset.split = "train"
        train_one_epoch(dataloader, optimizer, model, tokenizer, loss_fn, args)
        # save checkpoint
        # save_checkpoint(model, args.folder, epoch, args)
        dataset.split = "val"
        print("evaluating")
        # Jiachen TODO: update eval for your feature
        # eval(val_dataloader, model, tokenizer, args)


if __name__ == "__main__":
    main()
