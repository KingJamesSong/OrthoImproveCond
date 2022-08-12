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

python visualize.py  --nc_out 3 --sefa True --samples 6 --nz 30 --model_type gan128 --model_path ./checkpoints/celeba_nog/latest_netG.pth --save_dir ./nog &

wait
