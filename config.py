# Code based on https://github.com/tamarott/SinGAN
import argparse
import random
import numpy as np
import torch

from utils import set_seed


def get_arguments():
    parser = argparse.ArgumentParser()
    # Game Type - Supports: mario, mariokart
    parser.add_argument("--game", default="mario", help="Which game is to be used?")

    # workspace:
    parser.add_argument("--not_cuda", action="store_true", help="disables cuda", default=0)

    # load, input, save configurations:
    parser.add_argument("--netG", default="", help="path to netG (to continue training)")
    parser.add_argument("--netD1", default="", help="path to netD1 (to continue training)")
    parser.add_argument("--netD2", default="", help="path to netD2 (to continue training)")
    parser.add_argument("--manualSeed", type=int, help="manual seed")
    parser.add_argument("--out", help="output folder", default="./output")

    parser.add_argument("--input-dir", help="input image dir", default="input")
    parser.add_argument("--discriminator1-input-name", help="discriminator1 input image name", default="lvl_1-2.txt")
    parser.add_argument("--discriminator2-input-name", help="discriminator2 input image name", default="lvl_1-3.txt")

    # networks hyper parameters:
    parser.add_argument("--nfc", type=int, help="number of filters for conv layers", default=64)
    parser.add_argument("--num_layer", type=int, help="number of layers", default=3)

    parser.add_argument("--g_ker_size", type=int, help="generator kernel size", default=3)
    parser.add_argument("--g_padding_mode", type=str, help="padding mode for generator ('zeros', 'reflect', 'replicate' or 'circular')", default='replicate')
    parser.add_argument("--g_padding", type=int, help="padding amount for generator", default=0)
    parser.add_argument("--g_stride", type=int, help="stride for generator", default=1)

    parser.add_argument("--d_ker_size", type=int, help="discriminator kernel size", default=2)
    parser.add_argument("--d_padding_mode", type=str, help="padding mode for discriminator ('zeros', 'reflect', 'replicate' or 'circular')", default='replicate')
    parser.add_argument("--d_padding", type=int, help="padding amount for discriminator", default=0)
    parser.add_argument("--d_stride", type=int, help="stride for discriminator", default=2)

    parser.add_argument("--no_softmax", type=bool, help="disables generator softmax layer", default=False)

    # scaling parameters:
    parser.add_argument("--scales", nargs='+', type=float, help="Scales descending (< 1 and > 0)",
                        default=[0.88, 0.75, 0.5])
    parser.add_argument("--noise_update", type=float, help="additive noise weight", default=0.1)
    parser.add_argument("--pad_with_noise", type=bool, help="use reflection padding? (makes edges random)",
                        default=False)

    # optimization hyper parameters:
    parser.add_argument("--niter", type=int, default=4000, help="number of epochs to train per scale")
    parser.add_argument("--gamma", type=float, help="scheduler gamma", default=0.1)
    parser.add_argument("--lr_g", type=float, default=0.0005, help="learning rate, default=0.0005")
    parser.add_argument("--lr_d", type=float, default=0.0005, help="learning rate, default=0.0005")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
    parser.add_argument("--Gsteps", type=int, help="Generator inner steps", default=3)
    parser.add_argument("--Dsteps", type=int, help="Discriminator inner steps", default=3)
    parser.add_argument("--lambda_grad", type=float, help="gradient penalty weight", default=0.1)
    # alpha controls how much the reconstruction factors into the training. 0 = No reconstruction.
    parser.add_argument("--alpha", type=float, help="reconstruction loss weight", default=100)

    # possible token grouping (Experimental Feature! May break Everything!)
    parser.add_argument("--token_insert", type=int, help="layer in which token groupings will be split out "
                                                         "(<-2 means no grouping at all)", default=-2)
    
    # Alpha Layer Type for Multi-Discriminator
    parser.add_argument("--alpha_layer_type", type=str, help="alpha layer blending for multi-discriminators " 
                                                         "'half-and-half', 'all-zeros', or 'all-ones')", default="half-and-half")

    return parser


def post_config(opt):
    """ Initializes parameters. We're using Namespace opt to pass a lot of used parameters to many functions. """

    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    set_seed(opt.manualSeed)

    # Defaults for other namespace values that will be overwritten during runtime
    opt.nc_current = 12  # n tokens of level 1-1
    if not hasattr(opt, "out_"):
        opt.out_ = "%s/%s/" % (opt.out, opt.discriminator1_input_name[:-4])
    opt.outf = "0"  # changes with each scale trained
    opt.num_scales = len(opt.scales)  # number of scales is implicitly defined
    opt.noise_amp = 1.0  # noise amp for lowest scale always starts at 1
    opt.seed_road = None  # for mario kart seed roads after training
    opt.token_list = ['!', '#', '-', '1', '@', 'C', 'S', 'U', 'X', 'g', 'k', 't']  # default list of 1-1
    opt.ImgGen = []  # needs to be set to the correct image gen for each game
    opt.stop_scale = opt.num_scales  # which scale to stop on - usually always last scale defined

    return opt