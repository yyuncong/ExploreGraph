#!/bin/bash
#SBATCH --job-name=fsdp-stats
#SBATCH --output=eval_log/fsdp-dcs-%j.txt
#SBATCH --error=eval_log/fsdp-dcs-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

top_k_categories=10
echo "top_k_categories=$top_k_categories"

python singlegpu_stats.py \
--prefiltering \
--top_k_categories=$top_k_categories
