# OrthoImproveCond
ECCV22 paper "Improving Covariance Conditioning of the SVD Meta-layer by Orthogonality"

<img src="/Decorrelated BN/dbn_gradient.jpg" width="45%"><img src="/Decorrelated BN/dbn_lr.jpg" width="45%" hspace="0.3%">

We propose [nearest-orthogonal gradient (nog)](https://github.com/KingJamesSong/OrthoImproveCond/blob/main/Decorrelated%20BN/main_cifar100.py#L139) and [optimal learning rate (olr)](https://github.com/KingJamesSong/OrthoImproveCond/blob/main/Decorrelated%20BN/main_cifar100.py#L152) to enforce strict/relaxted orthogonality into the training of differentiable SVD layer, which can simultaneously improve the conditioning and generalization.

More extended experiments will be updated sooon! Stay tuned.

## Usage

Run decorrelated BN experiments with proposed techniques to improve covariance conditioning:

```python
CUDA_VISIBLE_DEVICES=0 python main_cifar100.py --norm='zcanormbatch' --batch_size=128 --nog --olr --ow
```

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

