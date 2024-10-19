# Code inspired by https://github.com/tamarott/SinGAN
import os

import torch
import wandb
from tqdm import tqdm

from mario.level_utils import encoded_to_ascii_level, token_to_group
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from mario.special_mario_downsampling import special_mario_downsampling
from mariokart.special_mariokart_downsampling import special_mariokart_downsampling
from models import init_models, reset_grads, restore_weights
from models.generator import Level_GeneratorConcatSkip2CleanAdd
from train_single_scale import train_single_scale


def train(generator_real, discriminator1_real, discriminator2_real, opt):
    """ Wrapper function for training. Calculates necessary scales then calls train_single_scale on each. """
    generators = []
    noise_maps = []
    noise_amplitudes = []

    if opt.game == 'mario':
        token_group = MARIO_TOKEN_GROUPS
    else:  # if opt.game == 'mariokart':
        token_group = MARIOKART_TOKEN_GROUPS

    scales = [[x, x] for x in opt.scales]
    opt.num_scales = len(scales)

    if opt.game == 'mario':
        generator_scaled_list = special_mario_downsampling(opt.num_scales, scales, generator_real, opt.token_list, opt.repr_type, opt.block2repr, opt.use_hierarchy)
        discriminator1_scaled_list = special_mario_downsampling(opt.num_scales, scales, discriminator1_real, opt.token_list, opt.repr_type, opt.block2repr, opt.use_hierarchy)
        discriminator2_scaled_list = special_mario_downsampling(opt.num_scales, scales, discriminator2_real, opt.token_list, opt.repr_type, opt.block2repr, opt.use_hierarchy)
    else:  # if opt.game == 'mariokart':
        scaled_list = special_mariokart_downsampling(opt.num_scales, scales, generator_real, opt.token_list)

    generator_reals = [*generator_scaled_list, generator_real]
    discriminator1_reals = [*discriminator1_scaled_list, discriminator1_real]
    discriminator2_reals = [*discriminator2_scaled_list, discriminator2_real]

    # If (experimental) token grouping feature is used:
    if opt.token_insert >= 0:
        reals = [(token_to_group(r, opt.token_list, token_group) if i < opt.token_insert else r) for i, r in enumerate(reals)]
        reals.insert(opt.token_insert, token_to_group(reals[opt.token_insert], opt.token_list, token_group))
    input_from_prev_scale = torch.zeros_like(generator_reals[0])

    # CDB HAX.
    stop_scale = 1
    #stop_scale = len(generator_reals)
    opt.stop_scale = stop_scale

    # Log the original input level as an image
    img = opt.ImgGen.render(encoded_to_ascii_level(generator_real, opt.token_list, opt.block2repr, opt.repr_type))
    wandb.log({"real": wandb.Image(img)}, commit=False)
    os.makedirs("%s/state_dicts" % (opt.out_), exist_ok=True)

    # Training Loop
    for current_scale in range(0, stop_scale):
        opt.outf = "%s/%d" % (opt.out_, current_scale)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        # If we are seeding, we need to adjust the number of channels
        if current_scale < (opt.token_insert + 1):  # (stop_scale - 1):
            opt.nc_current = len(token_group)

        # If using block2vec, our channels must be the dimension of the embedding
        if opt.repr_type == 'block2vec':
            opt.nc_current = generator_real.shape[1]

        # Initialize models
        D1, D2, G = init_models(opt)
        # If we are seeding, the weights after the seed need to be adjusted
        if current_scale == (opt.token_insert + 1):  # (stop_scale - 1):
            D1, D2, G = restore_weights(D1, D2, G, current_scale, opt)

        # Actually train the current scale
        z_opt, input_from_prev_scale, G = train_single_scale(D1, D2,  G, generator_reals, discriminator1_reals,
                                                             discriminator2_reals, generators, noise_maps,
                                                             input_from_prev_scale, noise_amplitudes, opt)

        # Reset grads and save current scale
        G = reset_grads(G, False)
        G.eval()
        D1 = reset_grads(D1, False)
        D1.eval()
        D2 = reset_grads(D2, False)
        D2.eval()

        generators.append(G)
        noise_maps.append(z_opt)
        noise_amplitudes.append(opt.noise_amp)

        torch.save(noise_maps, "%s/noise_maps.pth" % (opt.out_))
        torch.save(generators, "%s/generators.pth" % (opt.out_))
        torch.save(generator_reals, "%s/G_reals.pth" % (opt.out_))
        torch.save(discriminator1_reals, "%s/D1_reals.pth" % (opt.out_))
        torch.save(discriminator2_reals, "%s/D2_reals.pth" % (opt.out_))
        torch.save(noise_amplitudes, "%s/noise_amplitudes.pth" % (opt.out_))
        torch.save(opt.num_layer, "%s/num_layer.pth" % (opt.out_))
        torch.save(opt.token_list, "%s/token_list.pth" % (opt.out_))
        wandb.save("%s/*.pth" % opt.out_)

        torch.save(G.state_dict(), "%s/state_dicts/G_%d.pth" % (opt.out_, current_scale))
        wandb.save("%s/state_dicts/*.pth" % opt.out_)

        del D1, D2, G

    return generators, noise_maps, generator_reals, noise_amplitudes
