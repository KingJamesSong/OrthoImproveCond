import os
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms, utils
from model import Generator
from fid import get_fid_fn
from dataset import Dataset, infinite_loader
from augmentation import DiffAugment
from torch.utils.data import DataLoader
import cv2
import lpips

loss_fn = lpips.LPIPS(net='vgg')

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt",
        type=str,
        help="checkpoint file to continue training",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="path to the dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu/cuda",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./results",
        help="result save directory",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="num of samples",
    )
    parser.add_argument(
        "--traverse",
        action="store_true",
        help="traverse all eigen dimensions",
    )
    parser.add_argument(
        "--evaluate_FID",
        action="store_true",
        help="generate images to evaluate FID score",
    )
    parser.add_argument(
        "--evaluate_PPL",
        action="store_true",
        help="generate images to evaluate PPL score",
    )
    parser.add_argument(
        "--evaluate_VP",
        action="store_true",
        help="generate images to evaluate VP score",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="batch size",
    )
    args = parser.parse_args()

    device = args.device

    ckpt = torch.load(args.ckpt, map_location="cpu")

    train_args = ckpt["args"]

    g_ema = Generator(
        size=train_args.size,
        n_basis=train_args.n_basis,
        noise_dim=train_args.noise_dim,
        base_channels=train_args.base_channels,
        max_channels=train_args.max_channels,
    ).to(device).eval()
    g_ema.load_state_dict(ckpt["g_ema"])

    if args.traverse:
        print("traversing:")
        traverse_samples = 8
        traverse_range = 4.0
        intermediate_points = 9
        truncation = 0.7

        logdir = os.path.join(args.logdir, train_args.name, str(ckpt["step"]).zfill(7))
        os.makedirs(logdir)
        print(f"result path: {logdir}")

        with torch.no_grad():
            utils.save_image(
                g_ema.sample(args.n_sample),
                os.path.join(logdir, "sample.png"),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                value_range=(-1, 1),
            )

        es, zs = g_ema.sample_latents(traverse_samples, truncation=truncation)

        _, n_layers, n_dim = zs.shape

        offsets = np.linspace(-traverse_range, traverse_range, intermediate_points)
        for i_layer in range(n_layers):
            for i_dim in range(n_dim):
                print(f"  layer {i_layer} - dim {i_dim}")
                imgs = []
                for offset in offsets:
                    _zs = zs.clone()
                    _zs[:, i_layer, i_dim] = offset
                    with torch.no_grad():
                        img = g_ema((es, _zs)).cpu()
                        img = torch.cat([_img for _img in img], dim=1)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=2)

                imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
                Image.fromarray(imgs).save(
                    os.path.join(logdir, f"traverse_L{i_layer}_D{i_dim}.png")
                )

    if args.evaluate_FID:
        # data
        transform = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        dataset = Dataset(args.path, transform)
        loader = infinite_loader(
            DataLoader(
                dataset,
                batch_size=args.batch,
                shuffle=True,
                drop_last=True,
                num_workers=8,
            )
        )
        compute_fid = get_fid_fn(dataset, device=device)
        print("FID image generation evaluation")
        fid_score = compute_fid(g_ema)
        print(fid_score)
        #traverse_samples = 8
        #traverse_range = 4.0
        #intermediate_points = 9
        #truncation = 0.7
        #count = 0
        #logdir = "/data/ysong/generated_images/"
        #for index in range(0,5000):
        #    es, zs = g_ema.sample_latents(traverse_samples, truncation=truncation)
        #    _, n_layers, n_dim = zs.shape
        #    offsets = np.linspace(-traverse_range, traverse_range, intermediate_points)
        #    for i_layer in range(n_layers):
        #        for i_dim in range(n_dim):
        #            for offset in offsets:
        #                _zs = zs.clone()
        #                _zs[:, i_layer, i_dim] = offset
        #                with torch.no_grad():
        #                    img = g_ema((es, _zs)).cpu()
        #                    for _img in img:
        #                        _img = (_img.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
        #                        Image.fromarray(_img).save(os.path.join(logdir, f"{count}.png"))
        #                        count = count + 1
    if args.evaluate_VP:
        traverse_samples = 1
        traverse_range = 4.0
        intermediate_points = 9
        truncation = 0.7
        count = 0
        out_path = "/data/ysong/generated_images/"
        _, zs = g_ema.sample_latents(traverse_samples, truncation=truncation)
        latent_dim = zs.view(-1).size(0)
        for index in range(0, 10000):
            es, zs = g_ema.sample_latents(traverse_samples, truncation=truncation)
            _, n_layers, n_dim = zs.shape
            zs2 = zs.clone()
            z_1 = np.random.uniform(low=-4,
                                    high=4,
                                    size=[1, latent_dim])

            z_2 = np.random.uniform(low=-4,
                                    high=4,
                                    size=[1, latent_dim])

            idx = np.array(list(range(100)))  # full

            delta_dim = np.random.randint(0, latent_dim)
            delta_dim = idx[delta_dim]

            delta_onehot = np.zeros((1, latent_dim))
            delta_onehot[np.arange(delta_dim.size), delta_dim] = 1

            z_2 = np.where(delta_onehot > 0, z_2, z_1)

            delta_z = z_1 - z_2

            if index == 0:
                labels = delta_z
            else:
                labels = np.concatenate([labels, delta_z], axis=0)
            zs = torch.from_numpy(z_1).float().cuda().view(zs.shape)
            zs2 = torch.from_numpy(z_2).float().cuda().view(zs2.shape)
            with torch.no_grad():
                img1 = g_ema((es, zs)).squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                img2 = g_ema((es, zs2)).squeeze().cpu().detach().numpy().transpose((1, 2, 0))
            pair_np = np.concatenate([img1, img2], axis=1)
            img = (pair_np + 1) * 127.5
            cv2.imwrite(
                os.path.join(out_path,
                             'pair_%06d.jpg' % (index)), img)

        np.save(os.path.join(out_path, 'labels.npy'), labels)

    if args.evaluate_PPL:
        #ppl = compute_ppl(g_ema, num_samples=50000, epsilon=1e-4, space='z', sampling='end',
        #                                         crop=False, batch_size=2)
        #print(ppl)
        traverse_samples = 1
        traverse_range = 4.0
        intermediate_points = 9
        truncation = 0.7
        # Sampling loop.
        dist = []
        #g_ema.cpu()
        for index in range(0, 10000):
            es, zs = g_ema.sample_latents(traverse_samples, truncation=truncation)
            _, n_layers, n_dim = zs.shape
            z_1 = torch.rand(zs.shape, device=zs.device)
            z_2 = torch.rand(zs.shape, device=zs.device)
            t = torch.rand([z_2.shape[0]], device=z_2.device)
            zt0 = slerp(z_1, z_2, t.unsqueeze(1))
            zt1 = slerp(z_1, z_2, t.unsqueeze(1) + 1e-4)
            img1 = g_ema((es, zt0.view(zs.shape))).cpu().detach()
            img2 = g_ema((es, zt1.view(zs.shape))).cpu().detach()
            with torch.no_grad():
                d = loss_fn.forward(img1, img2)/ ( 1e-4 ** 2)
            dist.append(d)
            #print(d)
        dist = torch.cat(dist).cpu().numpy()
        lo = np.percentile(dist, 1, interpolation='lower')
        hi = np.percentile(dist, 99, interpolation='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
        print(float(ppl))

