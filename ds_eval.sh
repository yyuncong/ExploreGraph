#!/bin/bash
#SBATCH --job-name=eval-ds
#SBATCH --output=log/eval_dcs_ds-%j.txt
#SBATCH --error=log/eval_dcs_ds-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:6
#SBATCH --nodes=1
# activate the environment
# source /gpfs/u/home/LMCG/LMCGnngn/scratch/miniconda3x86/etc/profile.d/conda.sh
#source ~/.bashrc_dcs
#conda activate /gpfs/u/home/LMCG/LMCGnngn/scratch/miniconda3/envs/jc-eqa
#conda activate jc-eqa


python singlegpu_eval.py \
--folder ckpts/ds_new_data \
--random_permute \
--egocentric_views \
--lr=1e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 0

python singlegpu_eval.py \
--folder ckpts/ds_new_data \
--random_permute \
--egocentric_views \
--lr=1e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 1

python singlegpu_eval.py \
--folder ckpts/ds_new_data \
--random_permute \
--egocentric_views \
--lr=1e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 2

python singlegpu_eval.py \
--folder ckpts/ds_new_data \
--random_permute \
--egocentric_views \
--lr=1e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 3


python singlegpu_eval.py \
--folder ckpts/ds_new_data \
--random_permute \
--egocentric_views \
--lr=1e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 4



