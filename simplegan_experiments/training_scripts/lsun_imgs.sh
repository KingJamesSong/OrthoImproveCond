#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --signal=B:SIGTERM@120
#SBATCH -t 23:59:00
#SBATCH -c 10
#SBATCH --gres gpu:1
#SBATCH -o /data/ysong/dsprite.txt
#SBATCH --mem=16000

eval "$(conda shell.bash hook)"

bash

cd /nfs/data_chaos/ysong/simplegan_experiments/

conda activate latent

python gen_lsun.py --nz 30 --reg_type nog  --model gan128 --dataroot /nfs/data_chaos/ysong/lsun/church_outdoor_train_lmdb/ --dataset_mode lsun --name lsun_nog & #--continue_train --epoch_count 6 --name dsprites_nogconv &

wait