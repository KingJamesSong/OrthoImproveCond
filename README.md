# OrthoImproveCond
ECCV22 paper "Improving Covariance Conditioning of the SVD Meta-layer by Orthogonality"

More extended experiments will be updated sooon! Stay tuned.

## Usage

To run decorrelated BN experiments with 

```CUDA_VISIBLE_DEVICES=0 python main_cifar100.py --norm='zcanormbatch' --batch_size=128 --nog --olr --ow

## Citation

Please consider citing our paper if you think the code is helpful to your research.

```
@inproceedings{song2022fast,
  title={Improving Covariance Conditioning of the SVD Meta-layer by Orthogonality},
  author={Song, Yue and Sebe, Nicu and Wang, Wei},
  booktitle={ECCV},
  year={2022}
}
```

## Contact

If you have any questions or suggestions, please feel free to contact me

`yue.song@unitn.it`

