""" Main training script """

import argparse
import copy
import glob
import os
import random
import functools
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
#from dataset import ExploreDataset
#from dataset_snapshot import ExploreDataset
#from dataset_snapshot_ds import ExploreDataset
# from dataset_snapshot_tokens import ExploreDataset
# test exploration_data_clustering
from dataset_snapshot_tokens_new import ExploreDataset
# test visual_token_number = 16
# from dataset_snapshot_tokens_16 import ExploreDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from easydict import EasyDict
from accelerate import load_checkpoint_and_dispatch
import deepspeed
from peft import LoraConfig, get_peft_model

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
from monitor import log_gpu_memory_usage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %I:%M:%S",
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from tqdm import tqdm
import torch.nn.functional as F
import json
from loader import *

from deepspeed.utils import logger as deepspeed_logger
deepspeed_logger.setLevel(logging.WARN)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# TODO:
# 1. initialize lora config
# 2. use lora to wrap up the model

def train_one_epoch(dataloader, optimizer, model_engine, tokenizer, loss_fn, args):
    #print(type(llava_model))
    #print(type(llava_model.train()))
    # extract local rank and move data to corresponding device
    #llava_model = llava_model.train()
    model_engine.train()
    
    torch.cuda.empty_cache()
        
    pbar = tqdm(dataloader, disable=(args.rank != 0))
    total_combined_loss = 0
    total_filter_loss = 0
    total_selection_loss = 0
    total_sample = 0
    #random_generator = np.random.default_rng(seed = 10)
    #local_device = torch.device(f"cuda:{model_engine.local_rank}")
    max_sample_size = 0
    for sample in pbar:
        feature_dict = EasyDict(
            scene_feature=sample.scene_feature.to("cuda"),
            scene_insert_loc=sample.scene_insert_loc,
            scene_length=sample.scene_length,
        )
        input_ids = sample.input_ids.to("cuda")#.to("cpu")
        attention_mask = sample.attention_mask.to("cuda")#.to("cpu")
        max_sample_size = max(max_sample_size, input_ids.shape[1])
        '''
        print(f"Current max sample size {max_sample_size} in device {model_engine.local_rank}")
        print(tokenizer.decode(input_ids[0][input_ids[0] != tokenizer.pad_token_id]))
        '''
        labels = input_ids.clone()
        answer_indices = torch.where(labels == 22550)[1]

        for j, answer_idx in enumerate(answer_indices):
            labels[j, : answer_idx + 2] = -100

        labels[labels == tokenizer.pad_token_id] = -100
        #log_gpu_memory_usage(model_engine.local_rank,"after loading data")
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                feature_dict=feature_dict,
                output_hidden_states=True,
            )
        selection_loss = outputs.loss
        #log_gpu_memory_usage(model_engine.local_rank,"after selection forward pass")
        
        del outputs
        torch.cuda.empty_cache()
        #print("if the loss is kept", selection_loss)
        #log_gpu_memory_usage(model_engine.local_rank,"after remove unrelated cache")
        model_engine.backward(selection_loss)
        #log_gpu_memory_usage(model_engine.local_rank,"after selection backward")
        #total_selection_loss += selection_loss.item()
        #total_selection_sample += input_ids.shape[0]
        # Jiachen TODO: get the extra filter outputs with everything you added
        # and calculate the filter_loss and combine it with the total loss for training
        # Add the values of the two losses to the set_description line
        # None feature dict as a placeholder
        if args.prefiltering:
            filter_input_ids = sample.filter_input_ids.to("cuda")#.to("cpu")
            filter_attention_mask = sample.filter_attention_mask.to("cuda")#.to("cpu")
            filter_labels = filter_input_ids.clone()
            # choose the first answer as the separator
            filter_answer_indices = torch.where(filter_labels == 22550)[1]
            for j, answer_idx in enumerate(filter_answer_indices):
                filter_labels[j, : answer_idx + 2] = -100
            filter_labels[filter_labels == tokenizer.pad_token_id] = -100
            # optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                filter_outputs = model_engine(
                    input_ids=filter_input_ids,
                    attention_mask=filter_attention_mask,
                    labels=filter_labels,
                    feature_dict=None,
                    output_hidden_states=True,
                )
            filter_loss = filter_outputs.loss * args.filter_coeff
            #combined_loss += filter_loss
            del filter_outputs
            #log_gpu_memory_usage(model_engine.local_rank,"remove filter cache")
            model_engine.backward(filter_loss)
            #log_gpu_memory_usage(model_engine.local_rank,"after filter backward pass")
        #combined_loss.backward()
        #optimizer.step()
        #model_engine.backward(combined_loss)
        #log_gpu_memory_usage(model_engine.local_rank,"after backward")
        model_engine.step()
        total_selection_loss += selection_loss.item()*input_ids.shape[0]
        total_sample += input_ids.shape[0]
        if args.prefiltering:
            total_filter_loss += filter_loss.item()*input_ids.shape[0]
            pbar.set_description(
                f"loss: {(total_selection_loss + total_filter_loss) / total_sample:.3f}, selection_loss: {total_selection_loss / total_sample:.3f}, filter_loss: {total_filter_loss / total_sample:.3f}"
            )
        else:
            pbar.set_description(f"loss: {total_selection_loss / total_sample:.3f}")
        #log_gpu_memory_usage(model_engine.local_rank,"after one step")
    #print(f"max sample size {max_sample_size} in device {args.rank}")

def eval(dataloader, model, tokenizer, args):
    model.eval()
    total_combined_loss = 0
    total_selection_loss = 0
    total_filter_loss = 0
    total_sample = 0
    pbar = tqdm(dataloader, disable=(args.rank != 0))
    with torch.no_grad():
        for sample in pbar:
            # calculate selection loss
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
            combined_loss = selection_loss.item()
            # calculate filter loss
            if args.prefiltering:
                filter_input_ids = sample.filter_input_ids.to("cuda")
                filter_attention_mask = sample.filter_attention_mask.to("cuda")
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
                with torch.autocast(device_type="cuda"):
                    filter_outputs = model(
                        input_ids=filter_input_ids,
                        attention_mask=filter_attention_mask,
                        labels=filter_labels,
                        feature_dict=None,
                        output_hidden_states=True,
                    )
                filter_loss = filter_outputs.loss*args.filter_coeff
                combined_loss += filter_loss.item()
            total_combined_loss += combined_loss*input_ids.shape[0]
            total_selection_loss += selection_loss.item()*input_ids.shape[0]
            total_sample += input_ids.shape[0]
            if args.prefiltering:
                total_filter_loss += filter_loss.item()*input_ids.shape[0]
                pbar.set_description(
                    f"loss: {total_combined_loss / total_sample:.3f}, selection_loss: {total_selection_loss / total_sample:.3f}, filter_loss: {total_filter_loss / total_sample:.3f}"
                )
            else:
                pbar.set_description(f"loss: {total_combined_loss / total_sample:.3f}")
    if args.rank == 0:
        if args.prefiltering:
            print(
                f"loss: {(total_combined_loss) / total_sample:.3f}, selection_loss: {total_selection_loss / total_sample:.3f}, filter_loss: {total_filter_loss / total_sample:.3f}"
            )
        else:
            print(f"loss: {total_selection_loss / total_sample:.3f}")

def format_saving_folder(args):
    saving_folder = f"{args.folder}_{args.lr}_patch{args.patch_size}_ds"
    if args.add_positional_encodings:
        saving_folder += "_pos"
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
    if args.lora_enable:
        saving_folder += "_lora"
    return saving_folder

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
    parser.add_argument("--lr",default = 1e-6, type=float)
    parser.add_argument("--batch_size",default = 1, type = int)
    parser.add_argument("--folder", default="tmp", help="save folder")
    # argument for lora----------------------------------
    parser.add_argument("--lora_enable", default=False, action="store_true")
    parser.add_argument("--lora_r", default = 128, type=int)
    parser.add_argument("--lora_alpha", default = 256, type=int)
    parser.add_argument("--lora_dropout", default = 0.05, type=float)
    parser.add_argument("--lora_bias",default = "none", type=str)
    #----------------------------------------------------
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
    parser.add_argument("--filter_coeff", type=float, default=0.5)
    parser.add_argument(
        "--add_positional_encodings", action="store_true", default=False
    )
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument("--visual_feature_size", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=2048)
    # TODO: include deepspeed arguments here
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    #args = parser.parse_args()
    # set up random seed
    set_seed(args.seed)
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(device_id)
    print(f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}")
    # device_id = init_distributed_device(args)

    # wrap up the model with deepspeed
    #model_path = "liuhaotian/llava-v1.5-7b"
    model_path = "/gpfs/u/home/LMCG/LMCGhazh/scratch/external/yuncong/llava-v1.5-7b"
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, None, model_name, device_map = None, add_multisensory_token=True
    )
    '''
    print("start loading model")
    load_ds_checkpoint(model, "ds_tmp/checkpoint_0")
    print("successfully loaded model")
    '''
    #exit(0)
    #device_id = torch.device(f'cuda:{args.local_rank}')
    #torch.cuda.set_device(device_id)
    model = model.to('cpu')
    #log_gpu_memory_usage(args.local_rank,"after moving model to cpu")
    #print("if the model is correctly wraped?", type(model_engine))
    #print("local rank is", model_engine.local_rank)
    # Jiachen TODO: pass your parameter in the dataset file
    train_total_dataset = ExploreDataset(
        scene_path=args.scene_path,
        exploration_path=args.exploration_path,
        egocentric_views=args.egocentric_views,
        action_memory=args.action_memory,
        prefiltering=args.prefiltering,
        top_k_categories=args.top_k_categories,
        random_permute=args.random_permute,
        add_positional_encodings=args.add_positional_encodings,
        patch_size=args.patch_size,
        tokenizer=tokenizer,
        max_length=args.max_length,
        visual_feature_size=args.visual_feature_size
    )
    val_total_dataset = ExploreDataset(
        scene_path=args.scene_path,
        exploration_path=args.exploration_path,
        egocentric_views=args.egocentric_views,
        action_memory=args.action_memory,
        prefiltering=args.prefiltering,
        top_k_categories=args.top_k_categories,
        random_permute=args.random_permute,
        add_positional_encodings=args.add_positional_encodings,
        tokenizer=tokenizer,
        patch_size=args.patch_size,
        max_length=args.max_length,
        visual_feature_size=args.visual_feature_size,
        split="val",
    )
    train_index, test_index = train_total_dataset.split_index(test_ratio=0.999)
    train_dataset = Subset(train_total_dataset, train_index)
    val_dataset = Subset(val_total_dataset, test_index)
    
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        drop_last=False,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        pin_memory=True,
        num_workers=8,
        sampler=sampler,
        collate_fn=train_total_dataset.collate_wrapper,
    )
    '''
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        drop_last=False,
    )
    '''
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=3,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        collate_fn=val_total_dataset.collate_wrapper,
    )

    # wrap up the model with lora and deepspeed
    #print('if the lora is enabled', args.lora_enable)
    model.requires_grad_(True)
    del model.model.vision_tower
    lora_config = None
    if args.lora_enable:
        model, lora_config = lora_wrapper(model,args)
        # check trainable parameters
        # model.print_trainable_parameters()
        # compatiable with deepspeed config
        model.to(torch.float16)
    #log_gpu_memory_usage(args.local_rank,"after wrapping model with lora")
    #model.train()
    # TODO: initialize the deepspedd engine here
    # figure out where args/model_parameters come from
    # count the number of parameters that requires grad
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.lr
    )
    model, optimizer, _, _ = deepspeed.initialize(
        args = args,
        model = model,
        optimizer = optimizer
    )
        #model_parameters = [p for p in model.parameters() if p.requires_grad]) #model.parameters()
        #training_data = train_dataset,
        #collate_fn = train_total_dataset.collate_wrapper,
    #)
    # del the raw model
    # del model
    #log_gpu_memory_usage(args.local_rank,"after initializing deepspeed")
    #print("local rank is", model.local_rank)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # wrap model and optimizer with DDP

    #model = model.to("cpu")
    

    loss_fn = torch.nn.CrossEntropyLoss()
    # start training
    # may be use this to avoid out of memory
    saving_folder = format_saving_folder(args)
    for epoch in range(args.num_epochs):
        if args.rank == 0:
            print("Start training epoch %d" % epoch)
        # Jiachen TODO: update train_one_epoch for your feature
        #log_gpu_memory_usage(args.local_rank,"before training")
        train_one_epoch(dataloader, optimizer, model, tokenizer, loss_fn, args)
        # save checkpoint
        save_ds_checkpoint(model, saving_folder, epoch, args, lora_config, True)
        #print("evaluating")
        #break
        if args.rank == 0:
            print("Start evaluating epoch %d" % epoch)
        # Jiachen TODO: update eval for your feature
        eval(val_dataloader, model, tokenizer, args)
    

if __name__ == "__main__":
    main()
