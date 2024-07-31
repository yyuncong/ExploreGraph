import copy
import glob
import os
import functools
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
import json

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def lora_wrapper(model,args):
    if isinstance(args, dict):
        lora_config = LoraConfig(
            r = args['r'],
            lora_alpha = args['lora_alpha'],
            target_modules = args['target_modules'],
            lora_dropout = args['lora_dropout'],
            bias = args['bias'],
            task_type = args['task_type']
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type = 'CAUSAL_LM'
        )
    model = get_peft_model(model, lora_config)
    return model, lora_config.to_dict()

def load_checkpoint(model, args, name="checkpoint.pt"):
    checkpoint = torch.load(name, map_location="cpu")
    # wait until checkpoints in all processes are loaded
    torch.distributed.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint, True)
    del checkpoint
    torch.cuda.empty_cache()
    torch.distributed.barrier()

def load_ds_checkpoint(model,saving_folder, ckpt_indx, exclude_frozen_parameters = False):
    # this returns a model unwrapped lora
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    tag = "checkpoint_%d" % ckpt_indx
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        saving_folder, 
        tag, 
        exclude_frozen_parameters = exclude_frozen_parameters
    )
    if "lora_config.json" in os.listdir(os.path.join(saving_folder)):
        with open(os.path.join(saving_folder, "lora_config.json"), 'r') as f:
            lora_config = json.load(f)
        model, _ = lora_wrapper(model,lora_config)
    model.load_state_dict(state_dict, strict = False)
    return model


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

def save_ds_checkpoint(model_engine, folder, epoch, args, lora_config = None, exclude_frozen_parameters = False):
    try:
        if not os.path.exists(folder):
            os.mkdir(folder)
    except:
        pass
    #folder = os.path.join(folder, "checkpoint_%d" % epoch)
    model_engine.save_checkpoint(
        folder,
        tag = "checkpoint_%d" % epoch,
        exclude_frozen_parameters = exclude_frozen_parameters
    )
    if lora_config is not None and args.rank == 0:
        lora_config["target_modules"] = list(lora_config["target_modules"])
        with open(os.path.join(folder,"lora_config.json"), "w") as f:
            json.dump(lora_config, f)
    #return model_engine