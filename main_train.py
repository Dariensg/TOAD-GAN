from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name_1', help='input image name 1', required=True)
    parser.add_argument('--input_name_2', help='input image name 2', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt.output_name = opt.input_name_1[:-4] + "_" + opt.input_name_2[:-4]
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    d1_reals = []
    d2_reals = []
    g_reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    #if (os.path.exists(dir2save)):
    if False:
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        d1_real = functions.read_image(opt.input_name_1, opt)
        d2_real = functions.read_image(opt.input_name_2, opt)[:, :, :164, :244]

        # White G real for times without recon loss.
        g_real = torch.ones_like(d1_real)

        # 50/50 side to side
        #g_real = d1_real.clone()
        #g_real[:, :, :, 122:] = d2_real[:,:,:,122:]
        # Temp for testing one discrim on mixed image.
        #d1_real = g_real.clone()
        functions.adjust_scales2image(g_real, opt)
        train(opt, Gs, Zs, d1_reals, d2_reals, g_reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,g_reals,NoiseAmp,opt)
