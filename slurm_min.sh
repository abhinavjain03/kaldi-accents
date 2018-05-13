#!/bin/bash

#SBATCH --gres=gpu:3,gpu_mem:9000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=20            # number of CPU cores
#SBATCH --output=slurm-%j.out        # output file

./run_with_accent_embedding_with_ivectors_min.sh --affix bnf1024_accent_1024bnf_0.5_0.5 --train-stage -10