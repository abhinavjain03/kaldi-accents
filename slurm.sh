#!/bin/bash

#SBATCH --gres=gpu:3,gpu_mem:8000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=20            # number of CPU cores
#SBATCH --output=slurm-%j.out        # output file

./run_with_accent_embedding_with_ivectors_min.sh --affix bnf300_appended_multitask_correct_bignn_0.7_0.3 --train-stage 24
