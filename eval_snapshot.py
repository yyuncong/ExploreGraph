""" Main training script """

import argparse
import copy
import glob
import os
import random
import functools
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from eval_dataset_snapshot import (
    ExploreDataset,
    construct_selection_prompt,
    collate_selection_wrapper,
)
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
    checkpoint_path = os.path.join(args.ckpt_folder, args.folder, name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # wait until checkpoints in all processes are loaded
    # torch.distributed.barrier()
    # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    model.load_state_dict(checkpoint, True)
    del checkpoint
    # torch.cuda.empty_cache()
    # torch.distributed.barrier()


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


def eval(dataloader, model, tokenizer, args):
    model.eval()
    total = 0
    snapshot_gt_total = 0
    snapshot_type_correct = 0
    snapshot_id_correct = 0
    frontier_gt_total = 0
    frontier_type_correct = 0
    frontier_id_correct = 0

    # evaluation metric for pre-filtering
    # correctly answer: No object available
    ranking_empty_total = 0
    ranking_empty_correct = 0
    # the number of matched objects in answer and output
    ranking_match_total = 0
    ranking_match_correct = 0

    total_options = 0
    correct = 0
    total_filter_loss = 0
    total_selection_loss = 0
    total_sample = 0
    pbar = tqdm(dataloader)
    # pbar = tqdm(dataloader)
    with torch.no_grad():
        for sample in pbar:
            filter_input_ids = sample.filter_input_ids.to("cuda")
            filter_attention_mask = sample.filter_attention_mask.to("cuda")
            filter_labels = filter_input_ids.clone()
            filter_answer_indices = torch.where(filter_labels == 22550)[1]
            for j, answer_idx in enumerate(filter_answer_indices):
                filter_labels[j, : answer_idx + 2] = -100
            filter_labels[filter_labels == tokenizer.pad_token_id] = -100
            total_sample += filter_input_ids.shape[0]
            '''
            print(
                tokenizer.decode(
                    filter_input_ids[0][filter_input_ids[0] != tokenizer.pad_token_id]
                )
            )
            print(tokenizer.decode(filter_input_ids[0][filter_answer_indices[0] + 2 :]))
            '''
            # with torch.autocast(device_type="cuda"):
            #     filter_outputs = model(
            #         input_ids=filter_input_ids,
            #         attention_mask=filter_attention_mask,
            #         labels=filter_labels,
            #         feature_dict=None,
            #         output_hidden_states=True,
            #     )
            # filter_loss = filter_outputs.loss
            # total_filter_loss += filter_loss.item()
            # total_sample += filter_input_ids.shape[0]

            # we need filter input_ids length here to get the answer
            filter_input_ids = sample.filter_input_ids.to("cuda")
            filter_answer_ind = torch.where(sample.filter_input_ids == 22550)[1][
                0
            ].item()
            filter_end_ind = sample.filter_length[0].item() - 1
            filter_answer_ids = filter_input_ids[
                :, filter_answer_ind + 2 : filter_end_ind
            ]
            # print("filter answer (all)", filter_input_ids[:, filter_answer_ind + 2 :])
            filter_input_ids = filter_input_ids[:, : filter_answer_ind + 2]
            # print("filter answer", filter_answer_ids)

            with torch.inference_mode() and torch.autocast(device_type="cuda"):
                filter_output_ids = model.generate(
                    filter_input_ids,
                    feature_dict=None,
                    do_sample=False,
                    # allow for some long classes
                    max_new_tokens=100,
                )
            filter_outputs = (
                tokenizer.decode(filter_output_ids[0, filter_input_ids.shape[1] :])
                .replace("</s>", "")
                .strip()
            )
            filter_answer = (
                tokenizer.decode(filter_answer_ids[0]).replace("</s>", "").strip()
            )
            
            #print("the model output", filter_outputs)
            #print("decoded answer", filter_answer.replace("\n", "/"))
            
            if filter_answer == "No object available":
                ranking_empty_total += 1
                if filter_answer == filter_outputs:
                    ranking_empty_correct += 1
                filter_answer = []
            else:
                filter_outputs = filter_outputs.split("\n")
                filter_answer = filter_answer.split("\n")
                ranking_match_total += len(filter_answer)
                # the order matters
                for i in range(min(len(filter_outputs), len(filter_answer))):
                    if filter_outputs[i] == filter_answer[i]:
                        ranking_match_correct += 1
            #print("splited filter output", filter_outputs)
            #print("splited filter answer", filter_answer)
            # construct selection prompt and get the answer
            selection_dict = sample.selection_dict[0]
            selection_sample = construct_selection_prompt(
                tokenizer,
                selection_dict.scene_token_id,
                selection_dict.text_before_snapshot,
                selection_dict.feature_before_snapshot,
                selection_dict.frontier_text,
                selection_dict.frontier_features,
                selection_dict.frontier_prediction,
                selection_dict.snapshot_info_dict,
                True,
                # here the input ranking is the output of pre-filtering stage
                filter_outputs,
                args.top_k_categories,
                max_length=2048,
            )
            if isinstance(selection_sample, str):
                # Three different types of string indicating different problems
                # print(selection_sample)
                continue
            selection_sample = collate_selection_wrapper([selection_sample])
            feature_dict = EasyDict(
                scene_feature=selection_sample.scene_feature.to("cuda"),
                scene_insert_loc=selection_sample.scene_insert_loc,
                scene_length=selection_sample.scene_length,
            )
            input_ids = selection_sample.input_ids.to("cuda")
            # test the output of construct_selection_prompt
            '''
            print("selection prompt")
            print(
                tokenizer.decode(input_ids[0][input_ids[0] != tokenizer.pad_token_id])
            )
            '''
            
            length = torch.nonzero(input_ids).shape[0]
            decode_result = tokenizer.decode(input_ids[0][0: length])
            if '<unk>' in decode_result:
                print('unknow problem in tokenizer!')
            attention_mask = selection_sample.attention_mask.to("cuda")
            labels = input_ids.clone()
            answer_indices = torch.where(labels == 22550)[1]
            for j, answer_idx in enumerate(answer_indices):
                labels[j, : answer_idx + 2] = -100
            labels[labels == tokenizer.pad_token_id] = -100
            # with torch.autocast(device_type="cuda"):
            #     outputs = model(
            #         input_ids=input_ids,
            #         attention_mask=attention_mask,
            #         labels=labels,
            #         feature_dict=feature_dict,
            #         output_hidden_states=True,
            #     )
            # selection_loss = outputs.loss
            # total_selection_loss += selection_loss.item()

            input_ids = selection_sample.input_ids.to("cuda")
            answer_ind = torch.where(selection_sample.input_ids == 22550)[1][0].item()
            answer_ids = input_ids[:, answer_ind + 2 : answer_ind + 6]
            input_ids = input_ids[:, : answer_ind + 2]

            with torch.inference_mode() and torch.autocast(device_type="cuda"):
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
            #print("final selection result", outputs)
            gt = tokenizer.decode(answer_ids[0]).replace("</s>", "").strip()
            #print("ground truth", gt)
            gt_type = gt.split(" ")[0]
            gt_id = gt.split(" ")[1]
            outputs_type = outputs.split(" ")[0]
            outputs_id = outputs.split(" ")[1]
            if gt_type == "snapshot":
                snapshot_gt_total += 1
                if gt_type == outputs_type:
                    snapshot_type_correct += 1
                    if gt_id == outputs_id:
                        snapshot_id_correct += 1
            if gt_type == "frontier":
                frontier_gt_total += 1
                if gt_type == outputs_type:
                    frontier_type_correct += 1
                    if gt_id == outputs_id:
                        frontier_id_correct += 1

            total += 1
            if gt.lower().strip() == outputs.lower().strip():
                correct += 1

            pbar.set_description(f"acc: {correct / total_sample}")

    print("accuracy:", correct / total_sample)
    print("object type accuracy:", snapshot_type_correct / snapshot_gt_total)
    print("object id accuracy:", snapshot_id_correct / snapshot_gt_total)
    print("frontier type accuracy:", frontier_type_correct / frontier_gt_total)
    print("frontier id accuracy:", frontier_id_correct / frontier_gt_total)
    print("ranking empty accuracy:", ranking_empty_correct / ranking_empty_total)
    print("ranking match accuracy:", ranking_match_correct / ranking_match_total)
    # print("filter loss:", total_filter_loss / total_sample)
    # print("selection loss:", total_selection_loss / total_sample)
    print("frontiers total:", frontier_gt_total)
    print("objects total:", snapshot_gt_total)
    print("ranking empty total:", ranking_empty_total)
    print("ranking match total:", ranking_match_total)


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
    # parser.add_argument("--folder", default="tmp", help="save folder")
    # revise the saving folder for checkpoint
    parser.add_argument(
        "--ckpt_folder",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yuncong/ExploreGraph-dev/ckpts",
        help="save folder",
    )
    parser.add_argument(
        "--folder",
        # default="merged_16_1e-06_rand_filter_top10_coeff0.3_ego",
        default="tmp",
        help="save folder",
    )
    parser.add_argument("--ckpt_index", default=0, type=int)
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
    parser.add_argument("--prefiltering", action="store_true", default=False)
    parser.add_argument("--top_k_categories", type=int, default=5)
    parser.add_argument("--lr", default=1e-6, type=float)
    parser.add_argument(
        "--random_permute",
        action="store_true",
        help="if set true, randomly permute object/frontiers/pre-filtering classes",
        default=False,
    )
    parser.add_argument("--filter_coeff", type=float, default=0.5)
    parser.add_argument(
        "--add_positional_encodings", action="store_true", default=False
    )
    args = parser.parse_args()
    # args.local_rank, args.rank, args.world_size = world_info_from_env()
    # print(f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}")
    # device_id = init_distributed_device(args)

    # args.local_rank, args.rank, args.world_size = world_info_from_env()
    # print(f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}")

    model_path = "liuhaotian/llava-v1.5-7b"
    model_path = os.path.expanduser(model_path)
    print(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=None, add_multisensory_token=True
    )
    # train_total_dataset = ExploreDataset(
    #     scene_path=args.scene_path,
    #     exploration_path=args.exploration_path,
    #     egocentric_views=args.egocentric_views,
    #     action_memory=args.action_memory,
    #     tokenizer=tokenizer,
    #     max_length=2048,
    # )
    # print(train_total_dataset.scene_dir)
    val_total_dataset = ExploreDataset(
        scene_path=args.scene_path,
        exploration_path=args.exploration_path,
        egocentric_views=args.egocentric_views,
        action_memory=args.action_memory,
        prefiltering=args.prefiltering,
        top_k_categories=args.top_k_categories,
        add_positional_encodings=args.add_positional_encodings,
        tokenizer=tokenizer,
        max_length=2048,
        split="val",
    )
    # train_dataset, val_dataset = dataset, dataset
    # train_index, test_index = dataset.split_index(test_ratio=0.999)
    # train_dataset = Subset(dataset, train_index)
    # val_dataset = Subset(dataset, test_index)
    # distributed dataset
    train_index, test_index = val_total_dataset.split_index(test_ratio=0.999)
    # train_dataset, val_dataset = dataset, dataset
    # train_index, test_index = dataset.split_index(test_ratio=0.999)
    # train_dataset = Subset(train_total_dataset, train_index)
    val_dataset = Subset(val_total_dataset, test_index)
    # dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     pin_memory=True,
    #     num_workers=1,
    #     collate_fn=train_total_dataset.collate_wrapper,
    # )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        collate_fn=val_total_dataset.collate_wrapper,
    )

    # freeze model (only train the projector?)
    model.requires_grad_(False)
    # model.requires_grad_(True)
    del model.model.vision_tower

    # saving_folder = f"{args.folder}_{args.lr}"
    # dummy input for run
    saving_folder = f"{args.folder}_{args.lr}"
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
    print(saving_folder)
    args.folder = saving_folder

    load_checkpoint(model, args, name=f"checkpoint_{args.ckpt_index}.pt")
    model = model.float()

    model = model.to("cuda")

    # start training
    eval(val_dataloader, model, tokenizer, args)


if __name__ == "__main__":
    main()
