import argparse
import copy
import os
from tqdm import tqdm
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Discriminator
from dataset import Dataset, infinite_loader
from augmentation import DiffAugment
from loss import get_adversarial_losses, get_regularizer
from fid import get_fid_fn


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


if __name__ == "__main__":

    import os
    # There are some 0 byte JPG files that cause issues
    # I extracted 'data.tgz' into a folder called anime_faces
    # but you can put whatever folder your data is in instead.
    # This will go through all subfolders recursively.
    for root, dirs, files in os.walk("/data/ysong/anime-face"):
        for file in files:
            path = os.path.join(root, file)
            if os.stat(path).st_size == 0:
                print("Remove 0B size file:", path)
                os.remove(path)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="path to the dataset",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="ffhq",
        help="experiment name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu/cuda (does not support multi-GPU training for now)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="log root directory",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=10000,
        help="sample log period",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=10000,
        help="checkpoint save period",
    )
    parser.add_argument(
        "--fid",
        action="store_true",
        help="compute fid before saving checkpoints",
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
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="base learning rate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500000,
        help="train steps",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="num of log samples",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="num of workers for dataloader",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint file to continue training",
    )
    parser.add_argument(
        "--n_basis",
        type=int,
        default=6,
        help="subspace dimension for a generator layer",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=512,
        help="noise dimension for the input layer of the generator",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=16,
        help="num of base channels for generator/discriminator",
    )
    parser.add_argument(
        "--max_channels",
        type=int,
        default=512,
        help="max num of channels for generator/discriminator",
    )
    parser.add_argument(
        "--adv_loss",
        choices=["hinge", "non_saturating", "lsgan"],
        default="hinge",
        help="adversarial loss type",
    )
    parser.add_argument(
        "--orth_reg",
        type=float,
        default=100.0,
        help="basis orthogonality regularization weight",
    )
    parser.add_argument(
        "--d_reg",
        type=float,
        default=10.0,
        help="discriminator r1 regularization weight",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="discriminator lazy regularization period",
    )
    parser.add_argument(
        "--reg_type",
        type=str,
        default='nog',
        choices=["nog", "olr", "ol"],
        help="Orthogonal regularization (NOG|OLR|OL)",
    )
    args = parser.parse_args()

    device = args.device

    # models
    generator = Generator(
        size=args.size,
        n_basis=args.n_basis,
        noise_dim=args.noise_dim,
        base_channels=args.base_channels,
        max_channels=args.max_channels,
    ).to(device).train()
    g_ema = copy.deepcopy(generator).eval()

    discriminator = Discriminator(
        size=args.size,
        base_channels=args.base_channels,
        max_channels=args.max_channels,
    ).to(device).train()

    my_list1 = ['blocks.0.projection.U']
    my_list2 = ['blocks.1.projection.U']
    my_list3 = ['blocks.2.projection.U']
    my_list4 = ['blocks.3.projection.U']
    my_list5 = ['blocks.4.projection.U']
    params1 = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list1, generator.named_parameters()))))
    params2 = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list2, generator.named_parameters()))))
    params3 = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list3, generator.named_parameters()))))
    params4 = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list4, generator.named_parameters()))))
    params5 = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list5, generator.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in (my_list1+my_list2+my_list3+my_list4+my_list5), generator.named_parameters()))))
    # optimizers
    print(params1)
    print(params2)
    print(params3)
    print(params4)
    print(params5)
    for name, param in generator.named_parameters():
        if param.requires_grad:
            print(name)
    #print(generator)
    g_optim = torch.optim.Adam(
        [{'params': params1},{'params': params2},{'params': params3},{'params': params4},{'params': params5},{'params': base_params}],
        lr=args.lr,
        betas=(0.5, 0.99),
    )

    d_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.99),
    )

    start_step = 0
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        start_step = ckpt["step"]

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    # losses
    d_adv_loss_fn, g_adv_loss_fn = get_adversarial_losses(args.adv_loss)
    d_reg_loss_fn = get_regularizer("r1")

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
            num_workers=args.n_workers,
        )
    )

    # train utils
    augment = DiffAugment(policy='color,translation,cutout', p=0.6)
    ema = partial(accumulate, decay=0.5 ** (args.batch / (10 * 1000)))
    if args.fid:
        compute_fid = get_fid_fn(dataset, device=device)

    logdir = os.path.join(
        args.logdir, args.name,
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(os.path.join(logdir, "samples"))
    os.makedirs(os.path.join(logdir, "checkpoints"))
    tb_writer = SummaryWriter(logdir)
    log_sample = g_ema.sample_latents(args.n_sample)
    print(f"training log directory: {logdir}")

    # train loop
    for step in (iterator := tqdm(range(args.steps), initial=start_step)):

        step = step + start_step + 1
        if step > args.steps:
            break
        
        real = next(loader).to(device)

        # D
        with torch.no_grad():
            fake = generator.sample(args.batch)
        real_pred = discriminator(augment(real))
        fake_pred = discriminator(augment(fake))

        d_loss = d_adv_loss_fn(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if (step - start_step - 1)  % args.d_reg_every == 0:
            real.requires_grad = True
            real_pred = discriminator(augment(real))
            r1 = d_reg_loss_fn(real_pred, real) * args.d_reg

            discriminator.zero_grad()
            r1.backward()
            d_optim.step()

        # G
        fake = generator.sample(args.batch)
        fake_pred = discriminator(augment(fake))

        g_loss_adv = g_adv_loss_fn(fake_pred)

        if args.reg_type == 'ol':
            g_loss_reg = generator.orthogonal_regularizer() * args.orth_reg
            g_loss = g_loss_adv + g_loss_reg
        else:
            g_loss = g_loss_adv

        generator.zero_grad()
        g_loss.backward()
        # Nearest Orthogonal Gradient
        if args.reg_type == 'nog':
            generator.orthogonal_gradient()
        # Optimal Learning Rate
        if args.reg_type == 'olr':
            for i in range(0, 5):
                weight_grad = generator.blocks[i].projection.U.grad.view(-1, 1)
                weight = generator.blocks[i].projection.U.view(-1, 1)
                wtw = weight.t().mm(weight)
                ltw = weight_grad.t().mm(weight)
                ltl = weight_grad.t().mm(weight_grad)
                lr = float((wtw * ltw) / (wtw * ltl + 2 * ltw * ltw))
                if lr > 0 and lr < g_optim.param_groups[-1]['lr']:
                    g_optim.param_groups[i]['lr'] = lr
                else:
                    g_optim.param_groups[i]['lr'] = g_optim.param_groups[-1]['lr']
        g_optim.step()

        ema(g_ema, generator)

        # log
        iterator.set_description(
            f"d: {d_loss.item():.4f}; g: {g_loss.item():.4f} "
        )

        tb_writer.add_scalar("loss/D", d_loss.item(), step)
        tb_writer.add_scalar("loss/D_r1", r1.item(), step)
        tb_writer.add_scalar("loss/G", g_loss.item(), step)
        #tb_writer.add_scalar("loss/G_orth", g_loss_reg.item(), step)
        tb_writer.add_scalar("loss/G_adv", g_loss_adv.item(), step)

        if step % args.sample_every == 0:
            with torch.no_grad():
                utils.save_image(
                    g_ema(log_sample),
                    os.path.join(logdir, "samples", f"{str(step).zfill(7)}.png"),
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    value_range=(-1, 1),
                )

        if step % args.ckpt_every == 0:
            ckpt = {
                    "step": step,
                    "args": args,
                    "g": generator.state_dict(),
                    "d": discriminator.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
            }

            if args.fid:
                fid_score = compute_fid(g_ema)
                tb_writer.add_scalar("metric/FID", fid_score, step)
                ckpt["fid"] = fid_score

            torch.save(
                ckpt,
                os.path.join(logdir, "checkpoints", f"{str(step).zfill(7)}.pt"),
            )

