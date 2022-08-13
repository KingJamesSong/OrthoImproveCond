python train_ffhq.py DATA_ROOT  --size 256 --batch 64 --reg_type nog --name ffhq_nog &

wait

python test.py CKPT_ROOT --path DATA_ROOT --traverse --evaluate_FID --evaluate_VP --evaluate_PPL &

wait