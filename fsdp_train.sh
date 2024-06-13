#!/bin/bash
#SBATCH --job-name=test-fsdp-new
#SBATCH --output=log/%j.txt
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
# activate the environment
source /gpfs/u/home/LMCG/LMCGnngn/scratch/miniconda3x86/etc/profile.d/conda.sh
conda activate mllm



RANDOM=$$
DIV=1000
OFFSET=24000
MASTER_PORT=$(($RANDOM%$DIV+$OFFSET))
export TORCHELASTIC_LOG_LEVEL=DEBUG

#export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

# set environment variable 
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
NODE_RANK=${SLURM_PROCID}

# if ']' is the last character of the node list
SLURM=${SLURM_NODELIST:0:3}
if [ "${SLURM_NODELIST: -1}" == "]" ]; then
    if [ $SLURM == "npl" ]; then
        # NPL
        ip=${SLURM}${SLURM_NODELIST:4:2}
    else
        # DCS
        ip=${SLURM}${SLURM_NODELIST:4:3}
    fi
    FLAG=1
else
    ip=$SLURM_NODELIST
    FLAG=0
fi

if [ $SLURM == "dcs" ]; then
    # change to your local director
    export PYTHONUSERBASE="/gpfs/u/home/LMCG/LMCGnngn/scratch/.local"
fi

# print the ip address, node list and node rank
echo $ip
echo $SLURM_NODELIST
echo $NODE_RANK

# run the training script
NUM_GPUS_PER_NODE=${1:-8}
echo $NUM_GPUS_PER_NODE

if [ $FLAG -eq 1 ]; then
    NUM_NODES=${2:-2}
    echo $NUM_NODES
    CMD="torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --node_rank=$NODE_RANK
    --rdzv-id=$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$ip:$MASTER_PORT"
else
    CMD="torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port=$MASTER_PORT"
fi

echo $CMD
#torchrun
#    --nnodes=1:4
#    --nproc-per-node=$NUM_TRAINERS
#    --max-restarts=3
#    --rdzv-id=$JOB_ID
#    --rdzv-backend=c10d
#    --rdzv-endpoint=$HOST_NODE_ADDR
#    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

srun $CMD \
fsdp_train.py \
--folder retrieval_attention_7 \
--lr=1e-7 \
--num_epochs=8 \
--batch_size=1 \
--save_interval=1
