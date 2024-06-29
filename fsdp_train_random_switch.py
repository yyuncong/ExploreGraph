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
    FullStateDictConfig,  # config policy for full state actions
    CPUOffload,
    StateDictType,  # full state dict / local state dict
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, args, name="checkpoint.pt"):
    # may be we should consider rank here
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
    # offload_to_cpu: store the full state dict in cpu
    # rank0_only: avoid redundant saving
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if args.rank == 0:
        torch.save(cpu_state, name)
    torch.distributed.barrier()


def train_one_epoch(dataloader, optimizer, llava_model, tokenizer, loss_fn, args):
    llava_model = llava_model.train()
    pbar = tqdm(dataloader, disable=(args.rank != 0))
    total_combined_loss = 0
    total_filter_loss = 0
    total_selection_loss = 0
    total_filter_sample = 0
    total_selection_sample = 0

    random_generator = np.random.default_rng(seed=10)

    for sample in pbar:

        if random_generator.random() < 0.5:
            feature_dict = EasyDict(
                scene_feature=sample.scene_feature.to("cuda"),
                scene_insert_loc=sample.scene_insert_loc,
                scene_length=sample.scene_length,
            )
            input_ids = sample.input_ids.to("cuda")
            attention_mask = sample.attention_mask.to("cuda")
            labels = input_ids.clone()
            answer_indices = torch.where(labels == 22550)[1]

            for j, answer_idx in enumerate(answer_indices):
                labels[j, : answer_idx + 2] = -100

            labels[labels == tokenizer.pad_token_id] = -100
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                outputs = llava_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    feature_dict=feature_dict,
                    output_hidden_states=True,
                )

            selection_loss = outputs.loss
            combined_loss = selection_loss
            total_selection_loss += selection_loss.item()
            total_selection_sample += input_ids.shape[0]
        else:
            filter_input_ids = sample.filter_input_ids.to("cuda")
            filter_attention_mask = sample.filter_attention_mask.to("cuda")
            filter_labels = filter_input_ids.clone()
            # choose the first answer as the separator
            filter_answer_indices = torch.where(filter_labels == 22550)[1]
            for j, answer_idx in enumerate(filter_answer_indices):
                filter_labels[j, : answer_idx + 2] = -100
            filter_labels[filter_labels == tokenizer.pad_token_id] = -100
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                filter_outputs = llava_model(
                    input_ids=filter_input_ids,
                    attention_mask=filter_attention_mask,
                    labels=filter_labels,
                    feature_dict=None,
                    output_hidden_states=True,
                )
            filter_loss = filter_outputs.loss * args.filter_coeff
            combined_loss = filter_loss
            total_filter_loss += filter_loss.item()
            total_filter_sample += filter_input_ids.shape[0]

        combined_loss.backward()
        optimizer.step()

        pbar.set_description(
            f"loss: {(total_selection_loss + total_filter_loss) / (total_selection_sample + total_filter_sample):.3f}, selection_loss: {total_selection_loss / max(total_selection_sample, 1):.3f}, filter_loss: {total_filter_loss / max(total_filter_sample, 1):.3f}"
        )


def eval(dataloader, model, tokenizer, args):
    model.eval()
    total_combined_loss = 0
    total_selection_loss = 0
    total_filter_loss = 0
    total_sample = 0
    pbar = tqdm(dataloader, disable=(args.rank != 0))
    # pbar = tqdm(dataloader)
    with torch.no_grad():
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

            for j, answer_idx in enumerate(answer_indices):
                labels[j, : answer_idx + 2] = -100

            labels[labels == tokenizer.pad_token_id] = -100
            with torch.autocast(device_type="cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    feature_dict=feature_dict,
                    output_hidden_states=True,
                )
            selection_loss = outputs.loss
            
            filter_input_ids = sample.filter_input_ids.to("cuda")
            filter_attention_mask = sample.filter_attention_mask.to("cuda")
            filter_labels = filter_input_ids.clone()
            filter_answer_indices = torch.where(filter_labels == 22550)[1]
            for j, answer_idx in enumerate(filter_answer_indices):
                filter_labels[j, : answer_idx + 2] = -100
            filter_labels[filter_labels == tokenizer.pad_token_id] = -100

            with torch.autocast(device_type="cuda"):
                filter_outputs = model(
                    input_ids=filter_input_ids,
                    attention_mask=filter_attention_mask,
                    labels=filter_labels,
                    feature_dict=None,
                    output_hidden_states=True,
                )
            filter_loss = filter_outputs.loss
            # total_combined_loss += combined_loss.item()
            total_selection_loss += selection_loss.item()
            total_filter_loss += filter_loss.item()
            total_sample += input_ids.shape[0]
            pbar.set_description(
                f"loss: {(total_selection_loss + total_filter_loss) / total_sample:.3f}, selection_loss: {total_selection_loss / total_sample:.3f}, filter_loss: {total_filter_loss / total_sample:.3f}"
            )

    if args.rank == 0:
        print(
            f"loss: {(total_selection_loss + total_filter_loss) / total_sample:.3f}, selection_loss: {total_selection_loss / total_sample:.3f}, filter_loss: {total_filter_loss / total_sample:.3f}"
        )


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
        help="scene path",
    )
    parser.add_argument(
        "--exploration_path",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/",
        help="exploration path",
    )
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--save_interval", default=20, type=int)
    parser.add_argument("--eval_interval", default=2, type=int)
    parser.add_argument("--folder", default="tmp", help="save folder")
    parser.add_argument("--lr", default=1e-6, type=float)
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
    parser.add_argument("--prefiltering", action="store_true", default=False)
    parser.add_argument("--top_k_categories", type=int, default=5)
    parser.add_argument("--filter_coeff", type=float, default=0.5)
    args = parser.parse_args()

    # local rank: the rank within the node
    # rank(global rank): the rank cross nodes 0 ~ world_size - 1
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    print(
        f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}"
    )

    # assign different device id
    device_id = init_distributed_device(args)

    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )
    transformer_layer_cls = [
        LlamaDecoderLayer,
    ]

    # algorithm for model split
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
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
    train_total_dataset = ExploreDataset(
        scene_path=args.scene_path,
        exploration_path=args.exploration_path,
        egocentric_views=args.egocentric_views,
        action_memory=args.action_memory,
        prefiltering=args.prefiltering,
        top_k_categories=args.top_k_categories,
        random_permute=args.random_permute,
        tokenizer=tokenizer,
        max_length=2048,
    )
    val_total_dataset = ExploreDataset(
        scene_path=args.scene_path,
        exploration_path=args.exploration_path,
        egocentric_views=args.egocentric_views,
        action_memory=args.action_memory,
        prefiltering=args.prefiltering,
        top_k_categories=args.top_k_categories,
        random_permute=args.random_permute,
        tokenizer=tokenizer,
        max_length=2048,
        split="val",
    )
    train_index, test_index = train_total_dataset.split_index(test_ratio=0.999)
    # train_dataset, val_dataset = dataset, dataset
    # train_index, test_index = dataset.split_index(test_ratio=0.999)
    train_dataset = Subset(train_total_dataset, train_index)
    val_dataset = Subset(val_total_dataset, test_index)
    # distributed dataset
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        drop_last=False,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
        sampler=sampler,
        collate_fn=train_total_dataset.collate_wrapper,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        collate_fn=val_total_dataset.collate_wrapper,
    )

    # freeze model
    model.requires_grad_(True)
    # model.requires_grad_(False)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.model.mm_projector.requires_grad_(True)
    # model.model.layers[-1].requires_grad_(True)
    del model.model.vision_tower
    model.train()

    # ignored_modules = [
    #     model.model.mm_projector,
    #     model.model.layers[-1],
    # ]
    ignored_modules = []
    # split model to different sharding, each sharding a device
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=fpSixteen,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=ignored_modules,
        # use_orig_params=True,
    )
    model = model.to(device_id)
    if args.rank == 0:
        print(isinstance(model, FSDP))
        # print(isinstance(model.Embedding, FSDP))
        # print(isinstance(model.model.tactile_projector,FSDP))
        # print(model.__dict__)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    # start training

    saving_folder = f"{args.folder}_{args.lr}"
    if args.random_permute:
        saving_folder += "_rand"
    if args.prefiltering:
        saving_folder += "_filter"
        saving_folder += f"_top{args.top_k_categories}"
        saving_folder += f"_coeff{args.filter_coeff}"
    if args.egocentric_views:
        saving_folder += "_ego"
    if args.action_memory:
        saving_folder += "_mem"

    for epoch in range(args.num_epochs):
        if args.rank == 0:
            print("Start training epoch %d" % epoch)
        train_one_epoch(dataloader, optimizer, model, tokenizer, loss_fn, args)
        # # save checkpoint
        try:
            if epoch % args.save_interval == 0:
                save_checkpoint(model, saving_folder, epoch, args)
        except:
            pass
        try:
            eval(val_dataloader, model, tokenizer, args)
        except:
            pass


if __name__ == "__main__":
    main()
