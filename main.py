# Code inspired by https://github.com/tamarott/SinGAN
from generate_samples import generate_samples
from train import train
from mariokart.tokens import REPLACE_TOKENS as MARIOKART_REPLACE_TOKENS
from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from mariokart.special_mariokart_downsampling import special_mariokart_downsampling
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mario.special_mario_downsampling import special_mario_downsampling
from mario.level_utils import read_level, get_all_tokens, train_block2vec
from config import get_arguments, post_config
from loguru import logger
import wandb
import sys
import torch
import torch.nn.functional as F
from utils import get_discriminator1_scaling_tensor, get_discriminator2_scaling_tensor


def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    #return [opt.input_name.split(".")[0]]
    return [opt.d1_input_name.split(".")[0]]


def main():
    """ Main Training funtion. Parses inputs, inits logger, trains, and then generates some samples. """

    # torch.autograd.set_detect_anomaly(True)

    # Logger init
    logger.remove()
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      + "<level>{level}</level> | "
                      + "<light-black>{file.path}:{line}</light-black> | "
                      + "{message}")

    # Parse arguments
    opt = get_arguments().parse_args()
    opt = post_config(opt)

    # Init wandb
    run = wandb.init(mode="offline", project="mario", tags=get_tags(opt),
                     config=opt, dir=opt.out)
    opt.out_ = run.dir

    # Init game specific inputs
    replace_tokens = {}
    sprite_path = opt.game + '/sprites'
    if opt.game == 'mario':
        opt.ImgGen = MarioLevelGen(sprite_path)
        replace_tokens = MARIO_REPLACE_TOKENS
        downsample = special_mario_downsampling
    elif opt.game == 'mariokart':
        opt.ImgGen = MariokartLevelGen(sprite_path)
        replace_tokens = MARIOKART_REPLACE_TOKENS
        downsample = special_mariokart_downsampling
    else:
        NameError("name of --game not recognized. Supported: mario, mariokart")

    all_tokens = get_all_tokens(opt)

    block2repr = None
    if opt.repr_type == 'block2vec':
        block2repr = train_block2vec(opt, all_tokens)

    # Read level according to input arguments
    discriminator1_real = read_level(opt, opt.d1_input_name, block2repr, all_tokens, replace_tokens).to(opt.device)
    discriminator2_real = read_level(opt, opt.d2_input_name, block2repr, all_tokens, replace_tokens).to(opt.device)

    discriminator1_real_length = discriminator1_real.shape[-1]
    discriminator2_real_length = discriminator2_real.shape[-1]

    if discriminator1_real_length > discriminator2_real_length:
        # If D1 is longer, pad D2 right
        discriminator2_real = F.pad(discriminator2_real, (0, discriminator1_real_length - discriminator2_real_length, 0, 0), 'replicate')
    elif discriminator2_real_length > discriminator1_real_length:
        # If D2 is longer, pad D1 left
        discriminator1_real = F.pad(discriminator1_real, (discriminator1_real_length - discriminator1_real_length, 0, 0, 0), 'replicate')

    d1_chance = get_discriminator1_scaling_tensor(opt, discriminator1_real) / (get_discriminator1_scaling_tensor(opt, discriminator1_real) + get_discriminator2_scaling_tensor(opt, discriminator2_real))
    gen_lerp = torch.bernoulli(d1_chance)
    generator_real = discriminator2_real.lerp(discriminator1_real, gen_lerp).to(opt.device)

    # Train!
    generators, noise_maps, generator_reals, noise_amplitudes = train(generator_real, discriminator1_real, discriminator2_real, opt)

    # Generate Samples of same size as level
    logger.info("Finished training! Generating random samples...")
    in_s = None
    generate_samples(generators, noise_maps, generator_reals,
                     noise_amplitudes, opt, in_s=in_s)

    # Generate samples of smaller size than level
    #logger.info("Generating arbitrary sized random samples...")
    #scale_v = 0.8  # Arbitrarily chosen scales
    #scale_h = 0.4
    #real_down = downsample(1, [[scale_v, scale_h]], generator_real, opt.token_list, opt.repr_type, opt.block2repr, opt.use_hierarchy)
    #real_down = real_down[0]
    # necessary for correct input shape
    #in_s = torch.zeros(real_down.shape, device=opt.device)
    #generate_samples(generators, noise_maps, generator_reals, noise_amplitudes, opt, in_s=in_s,
    #                 scale_v=scale_v, scale_h=scale_h, save_dir="arbitrary_random_samples")


if __name__ == "__main__":
    main()
