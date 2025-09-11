export NCCL_P2P_DISABLE=1
export WANDB_KEY="fecb6fa17bef5ff2269bba21500e001f5101fb64"
export ENTITY="mon8c"
export PROJECT="dFlow"

current_time=$(date +"%m%d%H%M")

# Training
torchrun --nnodes=1 --nproc_per_node=4 scripts/main.py time=${current_time}

# Sampling (reusing the same time)
python scripts/sampling/sample_hydra.py time=${current_time}

# Evaluating (reusing the same time)
torchrun --nnodes=1 --nproc_per_node=4  scripts/evaluation/evaluate_hydra.py time=${current_time}