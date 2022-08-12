#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-mhug-staff
#SBATCH --signal=B:SIGTERM@120
#SBATCH -t 23:59:00
#SBATCH -c 10
#SBATCH --gres gpu:1
#SBATCH -o /data/ysong/dsprite.txt
#SBATCH --mem=16000

eval "$(conda shell.bash hook)"

bash

cd /data/ysong/simplegan_experiments/

conda activate latent

python train.py \
      --dataset_mode celeba \
      --model gan128 \
      --nz 30 \
      --reg_type nog \
      --dataroot /nfs/data_lambda/datasets/celeba/images \
      --name celeba_nog  \
      --save_latest_freq 10000 \
      --display_freq 20000 \
      --display_sample_freq 10000 \
      --print_freq 10000 &
      #--continue_train \
      #--epoch_count 54 &

wait