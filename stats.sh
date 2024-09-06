#!/bin/bash
#SBATCH --job-name=fsdp-stats
#SBATCH --output=eval_log/stats-%j.txt
#SBATCH --error=eval_log/stats-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

#top_k_categories=10
#echo "top_k_categories=$top_k_categories"

#python singlegpu_stats.py \
#--prefiltering \
#--top_k_categories=$top_k_categories

# max_length=2048

# echo "max_length=$max_length"

# python singlegpu_stats.py \
# --max_length=$max_length \
# --patch_size=1 \
# --visual_feature_size=4

max_length=4508

echo "max_length=$max_length"
python singlegpu_stats.py \
--max_length=$max_length \
--patch_size=1 \
--visual_feature_size=3 \









