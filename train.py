import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import transforms, utils
import lpips
import torchvision.models as models
from get_data import create_dataset, create_dataset_classes
from torch.utils.data import Dataset, DataLoader
from op import conv2d_gradfix
from model import Generator, Discriminator, Encoder


# Set requires_grad for model
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# Update weights of model1 with model2 and some decay rate
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

# Compute discriminator adversarial loss
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

# Compute discriminator regularization loss
def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

# Compute generator adversarial loss
def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

# Compute generator path regularization loss
def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def train(args, loader, generator, discriminator, encoder, classifier, g_optim, d_optim, e_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    mean_path_length = 0
    path_lengths = torch.tensor(0.0, device=device)

    requires_grad(classifier, False)
    l1_loss = nn.L1Loss()
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    kld_loss = nn.KLDivLoss(log_target=True, reduction='batchmean')

    # Decay factor for estimated mean average generator accumulation
    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.permute(0,3,1,2).float()
        real_img = real_img.to(device)

        # Train discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(encoder, False)

        fake_img, _, _ = generator(mapped_latents=[torch.cat((encoder(real_img), classifier(real_img)), dim=1)])

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        # Regularize discriminator
        if d_regularize:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        # Train generator and encoder
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(encoder, True)

        encoded_img = encoder(real_img)
        classified_img = classifier(real_img)
        fake_img, latents, _ = generator(mapped_latents=[torch.cat((encoded_img, classified_img), dim=1)], return_latents=True)

        fake_pred = discriminator(fake_img)

        g_loss = g_nonsaturating_loss(fake_pred)
        lpip = torch.mean(lpips_loss(fake_img, real_img))
        rec_loss = 0.1 * l1_loss(encoder(fake_img), encoded_img) + 0.1 * lpip + l1_loss(fake_img, real_img)
        cl_loss = kld_loss(nn.LogSoftmax(-1)(classifier(fake_img)), nn.LogSoftmax(-1)(classified_img))
        total_loss = g_loss + rec_loss + cl_loss

        generator.zero_grad()
        encoder.zero_grad()
        total_loss.backward()
        g_optim.step()
        e_optim.step()

        g_regularize = i % args.g_reg_every == 0
        # Regularize generator
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            encoded_img = encoder(real_img)
            classified_img = classifier(real_img)
            fake_img, latents, _ = generator(mapped_latents=[torch.cat((encoded_img, classified_img), dim=1)], return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

        # Update the estimated mean average of the model
        accumulate(g_ema, generator, accum)

        # Save a sample every 1000 iterations
        if i % 1000 == 0:
            with torch.no_grad():
                print(i)
                g_ema.eval()
                sample, _, _ = g_ema(mapped_latents=[torch.cat((encoded_img, classified_img), dim=1)])
                utils.save_image(
                    sample,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )

        # Save a checkpoint of the models every 10000 iterations
        if i % 10000 == 0:
            torch.save(
                {
                    "g": generator.state_dict(),
                    "d": discriminator.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "e_optim": e_optim.state_dict(),
                    "args": args,
                    "e":encoder.state_dict(),
                },
                f"checkpoint/{str(i).zfill(6)}.pt",
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument(
        "--iter", type=int, default=250000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--size", type=int, default=64, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=1,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--dir", type=str, default='$TMPDIR', help="directory of training data"
    )
    args = parser.parse_args()

    args.latent = 514
    args.n_mlp = 8
    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    encoder = Encoder(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr)

    # Load a checkpoint
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        encoder.load_state_dict(ckpt['e'])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        e_optim.load_state_dict(ckpt["e_optim"])

    # train_path = args.dir + '/afhq/train/'
    # train_data = create_dataset_classes(train_path, 64, ['cat', 'dog'])
    # train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)

    train_path_thumb = args.dir + '/ffhq/train/'
    train_data_thumb = create_dataset(train_path_thumb, 64)
    train_loader = DataLoader(train_data_thumb, batch_size=4, shuffle=True)

    classifier = models.mobilenet_v2(pretrained=False, num_classes=2)
    classifier.load_state_dict(torch.load("classifier_model_celeba.pt"))
    classifier.to(device)
    classifier.eval()

    train(args, train_loader, generator, discriminator, encoder, classifier, g_optim, d_optim, e_optim, g_ema, device)
