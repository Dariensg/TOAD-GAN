import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import interpolate
from loguru import logger
from tqdm import tqdm
from utils import get_discriminator1_scaling_tensor, get_discriminator2_scaling_tensor

import wandb

from draw_concat import draw_concat
from generate_noise import generate_spatial_noise
from mario.level_utils import group_to_token, one_hot_to_ascii_level, token_to_group
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from models import calc_gradient_penalty, save_networks


def update_noise_amplitude(z_prev, real, opt):
    """ Update the amplitude of the noise for the current scale according to the previous noise map. """
    RMSE = torch.sqrt(F.mse_loss(real, z_prev))
    return opt.noise_update * RMSE


def train_single_scale(D1, D2, G, generator_reals, discriminator1_reals, discriminator2_reals, generators, noise_maps, input_from_prev_scale, noise_amplitudes, opt):
    """ Train one scale. D and G are the current discriminator and generator, reals are the scaled versions of the
    original level, generators and noise_maps contain information from previous scales and will receive information in
    this scale, input_from_previous_scale holds the noise map and images from the previous scale, noise_amplitudes hold
    the amplitudes for the noise in all the scales. opt is a namespace that holds all necessary parameters. """
    current_scale = len(generators)
    generator_real = generator_reals[current_scale]
    discriminator1_real = discriminator1_reals[current_scale]
    discriminator2_real = discriminator2_reals[current_scale]

    if opt.game == 'mario':
        token_group = MARIO_TOKEN_GROUPS
    else:  # if opt.game == 'mariokart':
        token_group = MARIOKART_TOKEN_GROUPS

    nzx = generator_real.shape[2]  # Noise size x
    nzy = generator_real.shape[3]  # Noise size y

    padsize = int(1 * opt.num_layer)  # As kernel size is always 3 currently, padsize goes up by one per layer

    upsample = nn.Upsample(size=(nzx,nzy), mode='nearest')

    if not opt.pad_with_noise:
        pad_noise = nn.ZeroPad2d(padsize)
        pad_image = nn.ZeroPad2d(padsize)
    else:
        pad_noise = nn.ReflectionPad2d(padsize)
        pad_image = nn.ReflectionPad2d(padsize)

    # setup optimizer
    optimizerD1 = optim.Adam(D1.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerD2 = optim.Adam(D2.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD1, milestones=[1600, 2500], gamma=opt.gamma)
    schedulerD2 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD2, milestones=[1600, 2500], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600, 2500], gamma=opt.gamma)

    if current_scale == 0:  # Generate new noise
        z_opt = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device)
        z_opt = pad_noise(z_opt)
    else:  # Add noise to previous output
        z_opt = torch.zeros([1, opt.nc_current, nzx, nzy]).to(opt.device)
        z_opt = pad_noise(z_opt)

    logger.info("Training at scale {}", current_scale)
    for epoch in tqdm(range(opt.niter)):
        step = current_scale * opt.niter + epoch
        noise_ = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device)
        noise_ = pad_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            
            # generate fake
            if (j == 0) & (epoch == 0):
                if current_scale == 0:  # If we are in the lowest scale, noise is generated from scratch
                    prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                    input_from_prev_scale = prev
                    prev = pad_image(prev)
                    z_prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                    z_prev = pad_noise(z_prev)
                    opt.noise_amp = 1
                else:  # First step in NOT the lowest scale
                    # We need to adapt our inputs from the previous scale and add noise to it
                    prev = draw_concat(generators, noise_maps, generator_reals, noise_amplitudes, input_from_prev_scale,
                                       "rand", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        prev = group_to_token(prev, opt.token_list, token_group)

                    prev = interpolate(prev, generator_real.shape[-2:], mode="bilinear", align_corners=False)
                    prev = pad_image(prev)
                    z_prev = draw_concat(generators, noise_maps, generator_reals, noise_amplitudes, input_from_prev_scale,
                                         "rec", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        z_prev = group_to_token(z_prev, opt.token_list, token_group)

                    z_prev = interpolate(z_prev, generator_real.shape[-2:], mode="bilinear", align_corners=False)
                    opt.noise_amp = update_noise_amplitude(z_prev, generator_real, opt)
                    z_prev = pad_image(z_prev)
            else:  # Any other step
                prev = draw_concat(generators, noise_maps, generator_reals, noise_amplitudes, input_from_prev_scale,
                                   "rand", pad_noise, pad_image, opt)

                # For the seeding experiment, we need to transform from token_groups to the actual token
                if current_scale == (opt.token_insert + 1):
                    prev = group_to_token(prev, opt.token_list, token_group)

                prev = interpolate(prev, generator_real.shape[-2:], mode="bilinear", align_corners=False)
                prev = pad_image(prev)

            # After creating our correct noise input, we feed it to the generator:
            noise = opt.noise_amp * noise_ + prev
            fake = G(noise.detach(), prev, temperature=1 if current_scale != opt.token_insert else 1)

            # === D1 ===

            # train with real
            D1.zero_grad()

            output_D1_real = D1(discriminator1_real).to(opt.device)

            errD1_real = -output_D1_real
            errD1_real.backward(gradient=torch.ones_like(errD1_real), retain_graph=True)

            # Then run the result through the discriminator
            output_D1_fake = D1(fake.detach())
            errD1_fake = output_D1_fake * get_discriminator1_scaling_tensor(opt, output_D1_real)

            # Backpropagation
            errD1_fake.backward(gradient=torch.ones_like(errD1_fake), retain_graph=False)

            # Gradient Penalty
            D1_gradient_penalty = calc_gradient_penalty(opt, D1, get_discriminator1_scaling_tensor, discriminator1_real, fake, opt.lambda_grad, opt.device)
            D1_gradient_penalty.backward(retain_graph=False)

            # Logging:
            if step % 10 == 0:
                wandb.log({f"D1(G(z))@{current_scale}": errD1_fake.mean().item(),
                           f"D1(x)@{current_scale}": -errD1_real.mean().item(),
                           f"D1_gradient_penalty@{current_scale}": D1_gradient_penalty.item()
                           },
                          step=step, sync=False)
            optimizerD1.step()

            # === D2 ===

            # train with real
            D2.zero_grad()

            output_D2_real = D2(discriminator2_real).to(opt.device)

            errD2_real = -output_D2_real
            errD2_real.backward(gradient=torch.ones_like(errD2_real), retain_graph=True)

            # Then run the result through the discriminator
            output_D2_fake = D2(fake.detach())
            errD2_fake = output_D2_fake * get_discriminator2_scaling_tensor(opt, output_D2_real)

            # Backpropagation
            errD2_fake.backward(gradient=torch.ones_like(errD2_fake), retain_graph=False)

            # Gradient Penalty
            D2_gradient_penalty = calc_gradient_penalty(opt, D2, get_discriminator2_scaling_tensor, discriminator2_real, fake, opt.lambda_grad, opt.device)
            D2_gradient_penalty.backward(retain_graph=False)

            # Logging:
            if step % 10 == 0:
                wandb.log({f"D2(G(z))@{current_scale}": errD2_fake.mean().item(),
                           f"D2(x)@{current_scale}": -errD2_real.mean().item(),
                           f"D2_gradient_penalty@{current_scale}": D2_gradient_penalty.item()
                           },
                          step=step, sync=False)
            optimizerD2.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            G.zero_grad()
            fake = G(noise.detach(), prev.detach(), temperature=1 if current_scale != opt.token_insert else 1)
            output_D1_G = D1(fake)
            output_D2_G = D2(fake)

            output_D1_G = output_D1_G * get_discriminator1_scaling_tensor(opt, output_D1_G)
            output_D2_G = output_D2_G * get_discriminator2_scaling_tensor(opt, output_D2_G)

            errD1_G = -output_D1_G
            errD2_G = -output_D2_G

            # Error needs to match fake shape (minus batch dim if any).
            errD1_tensor = upsample(errD1_G).squeeze().expand(fake.shape)
            errD2_tensor = upsample(errD2_G).squeeze().expand(fake.shape)
            combined_error = errD1_tensor + errD2_tensor
            combined_error.backward(gradient=torch.ones_like(combined_error))

            if opt.alpha != 0:  # i. e. we are trying to find an exact recreation of our input in the lat space
                Z_opt = opt.noise_amp * z_opt + z_prev
                G_rec = G(Z_opt.detach(), z_prev, temperature=1 if current_scale != opt.token_insert else 1)
                rec_loss = opt.alpha * F.mse_loss(G_rec, generator_real)
                rec_loss.backward(retain_graph=False)  # TODO: Check for unexpected argument retain_graph=True
                rec_loss = rec_loss.detach()
            else:  # We are not trying to find an exact recreation
                rec_loss = torch.zeros([])
                Z_opt = z_opt

            optimizerG.step()

        # More Logging:
        if step % 10 == 0:
            wandb.log({f"noise_amplitude@{current_scale}": opt.noise_amp,
                       f"rec_loss@{current_scale}": rec_loss.item()},
                      step=step, sync=False, commit=True)

        # Rendering and logging images of levels
        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            if opt.token_insert >= 0 and opt.nc_current == len(token_group):
                token_list = [list(group.keys())[0] for group in token_group]
            else:
                token_list = opt.token_list

            img = opt.ImgGen.render(one_hot_to_ascii_level(fake.detach(), token_list))
            img2 = opt.ImgGen.render(one_hot_to_ascii_level(
                G(Z_opt.detach(), z_prev, temperature=1 if current_scale != opt.token_insert else 1).detach(),
                token_list))
            
            generator_real_scaled = one_hot_to_ascii_level(generator_real.detach(), token_list)
            generator_img3 = opt.ImgGen.render(generator_real_scaled)

            discriminator1_real_scaled = one_hot_to_ascii_level(discriminator1_real.detach(), token_list)
            discriminator1_img3 = opt.ImgGen.render(discriminator1_real_scaled)

            discriminator2_real_scaled = one_hot_to_ascii_level(discriminator2_real.detach(), token_list)
            discriminator2_img3 = opt.ImgGen.render(discriminator2_real_scaled)

            wandb.log({f"G(z)@{current_scale}": wandb.Image(img),
                       f"G(z_opt)@{current_scale}": wandb.Image(img2),
                       f"generator_real@{current_scale}": wandb.Image(generator_img3),
                       f"discriminator1_real@{current_scale}": wandb.Image(discriminator1_img3),
                       f"discriminator2_real@{current_scale}": wandb.Image(discriminator2_img3)},
                      sync=False, commit=False)

            generator_real_scaled_path = os.path.join(wandb.run.dir, f"generator_real@{current_scale}.txt")
            with open(generator_real_scaled_path, "w") as f:
                f.writelines(generator_real_scaled)
            wandb.save(generator_real_scaled_path)

            discriminator1_real_scaled_path = os.path.join(wandb.run.dir, f"discriminator1_real@{current_scale}.txt")
            with open(discriminator1_real_scaled_path, "w") as f:
                f.writelines(discriminator1_real_scaled)
            wandb.save(discriminator1_real_scaled_path)

            discriminator2_real_scaled_path = os.path.join(wandb.run.dir, f"discriminator2_real@{current_scale}.txt")
            with open(discriminator2_real_scaled_path, "w") as f:
                f.writelines(discriminator2_real_scaled)
            wandb.save(discriminator2_real_scaled_path)

        # Learning Rate scheduler step
        schedulerD1.step()
        schedulerD2.step()
        schedulerG.step()

    # Save networks
    torch.save(z_opt, "%s/z_opt.pth" % opt.outf)
    save_networks(G, D1, D2, z_opt, opt)
    wandb.save(opt.outf)
    return z_opt, input_from_prev_scale, G
