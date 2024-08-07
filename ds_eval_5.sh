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
--folder ckpts/ds_new \
--random_permute \
--egocentric_views \
--lr=2e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 5

python singlegpu_eval.py \
--folder ckpts/ds_new \
--random_permute \
--egocentric_views \
--lr=2e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 6

python singlegpu_eval.py \
--folder ckpts/ds_new \
--random_permute \
--egocentric_views \
--lr=2e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 7

python singlegpu_eval.py \
--folder ckpts/ds_new \
--random_permute \
--egocentric_views \
--lr=2e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 8

python singlegpu_eval.py \
--folder ckpts/ds_new \
--random_permute \
--egocentric_views \
--lr=2e-6 \
--patch_size=2 \
--lora_enable \
--deepspeed_enable \
--ckpt_index 9


