#!/bin/bash

#SBATCH --gres=gpu:3,gpu_mem:1000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=20            # number of CPU cores
#SBATCH --output=slurm-%j.out        # output file

./run_with_accent_embedding_with_ivectors.sh --affix bnf300_appended_cv_train_nz --train-stage -10