python train_animeface.py DATA_ROOT  --size 128 --batch 64 --reg_type nog --name anime_nog &

wait

python test.py CKPT_ROOT --path DATA_ROOT --traverse --evaluate_FID --evaluate_VP --evaluate_PPL &

wait