# OrthoImproveCond
ECCV22 paper ["Improving Covariance Conditioning of the SVD Meta-layer by Orthogonality"](https://arxiv.org/pdf/2207.02119.pdf)

<img src="/Decorrelated BN/dbn_gradient.jpg" width="45%"><img src="/Decorrelated BN/dbn_lr.jpg" width="45%" hspace="0.3%">

We propose [nearest-orthogonal gradient (nog)](https://github.com/KingJamesSong/OrthoImproveCond/blob/main/Decorrelated%20BN/main_cifar100.py#L139) and [optimal learning rate (olr)](https://github.com/KingJamesSong/OrthoImproveCond/blob/main/Decorrelated%20BN/main_cifar100.py#L152) to enforce strict/relaxed orthogonality into the training of differentiable SVD layer, which can simultaneously improve the conditioning and generalization. The combination with [orthogonal convolution](https://github.com/KingJamesSong/OrthoImproveCond/blob/main/Decorrelated%20BN/models/skew_symmetric_conv.py#L12) could further boost the performance.

<img src="ffhq_finegrained.jpg" width="99%">

<img src="celeba_comparison2.jpg" width="99%">

The proposed orthogonality techniques can be also used for unsupervised latent disentanglement of generative models.


More extended experiments will be updated sooon! Stay tuned.

## Usage (decorrelated BN)

Run decorrelated BN experiments with proposed techniques to improve covariance conditioning:

```python
CUDA_VISIBLE_DEVICES=0 python main_cifar100.py --norm='zcanormbatch' --batch_size=128 --nog --olr --ow
```

## Usage (Orthogonal EigenGAN)

## Usage (Orthogonal vanilla/simple GAN)

```python
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_mode celeba --model gan128 --nz 30 --reg_type nog --dataroot CELEBA_ROOT --name celeba_nog  
```


## Requirements

Check [latent.yml](https://github.com/KingJamesSong/OrthoImproveCond/blob/main/latent.yml) for the full list of required packages.

## Citation

Please consider citing our paper if you think the code is helpful to your research.

```
@inproceedings{song2022improving,
  title={Improving Covariance Conditioning of the SVD Meta-layer by Orthogonality},
  author={Song, Yue and Sebe, Nicu and Wang, Wei},
  booktitle={ECCV},
  year={2022}
}
```

## Contact

If you have any questions or suggestions, please feel free to contact me

`yue.song@unitn.it`

