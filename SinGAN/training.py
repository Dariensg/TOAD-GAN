import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

def train(opt,Gs,Zs,d1_reals,d2_reals,g_reals,NoiseAmp):
    d1_real_ = functions.read_image(opt.input_name_1, opt)
    d2_real_ = functions.read_image(opt.input_name_2, opt)[:, :, :164, :244]

    # White G real for testing without recon loss.
    g_real_ = torch.ones_like(d1_real_)

    # 50/50 G real
    #g_real_ = d1_real_.clone()
    #g_real_[:, :, :, 122:] = d2_real_[:,:,:,122:]
    # Temp for testing one discrim on mixed image.
    #d1_real_ = g_real_.clone()
    in_s = 0
    scale_num = 0
    d1_real = imresize(d1_real_,opt.scale1,opt)
    d1_reals = functions.creat_reals_pyramid(d1_real,d1_reals,opt)
    d2_real = imresize(d2_real_,opt.scale1,opt)
    d2_reals = functions.creat_reals_pyramid(d2_real,d2_reals,opt)
    g_real = imresize(g_real_,opt.scale1,opt)
    g_reals = functions.creat_reals_pyramid(g_real,g_reals,opt)
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/d1_real_scale.png' %  (opt.outf), functions.convert_image_np(d1_reals[scale_num]), vmin=0, vmax=1)
        plt.imsave('%s/d2_real_scale.png' %  (opt.outf), functions.convert_image_np(d2_reals[scale_num]), vmin=0, vmax=1)
        plt.imsave('%s/g_real_scale.png' %  (opt.outf), functions.convert_image_np(g_reals[scale_num]), vmin=0, vmax=1)

        D1_curr,D2_curr,G_curr = init_models(opt)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D1_curr.load_state_dict(torch.load('%s/%d/netD1.pth' % (opt.out_,scale_num-1)))
            D2_curr.load_state_dict(torch.load('%s/%d/netD2.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D1_curr,D2_curr,G_curr,d1_reals,d2_reals,g_reals,Gs,Zs,in_s,NoiseAmp,opt)

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D1_curr = functions.reset_grads(D1_curr,False)
        D1_curr.eval()
        D2_curr = functions.reset_grads(D2_curr,False)
        D2_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(d1_reals, '%s/d1_reals.pth' % (opt.out_))
        torch.save(d2_reals, '%s/d2_reals.pth' % (opt.out_))
        torch.save(g_reals, '%s/g_reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D1_curr,D2_curr,G_curr
    return



def train_single_scale(netD1,netD2,netG,d1_reals,d2_reals,g_reals,Gs,Zs,in_s,NoiseAmp,opt,centers=None):

    d1_scale = None
    d2_scale = None

    d1_real = d1_reals[len(Gs)]
    d2_real = d2_reals[len(Gs)]
    g_real = g_reals[len(Gs)]
    opt.nzx = g_real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = g_real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = g_real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = g_real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0.0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD1 = optim.Adam(netD1.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerD2 = optim.Adam(netD2.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD1,milestones=[1600],gamma=opt.gamma)
    schedulerD2 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD2,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD1plot = []
    errD2plot = []
    errGplot = []
    D1_real2plot = []
    D1_fake2plot = []
    D2_real2plot = []
    D2_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD1.zero_grad()
            netD2.zero_grad()

            output1 = netD1(d1_real).to(opt.device)
            output2 = netD2(d2_real).to(opt.device)

            if d1_scale is None:
                r = output1.shape[-2]
                c = output1.shape[-1]

                # Make the scaling template
                d1_templ = torch.tensor([[1]], dtype=torch.float).to(opt.device)
                d2_templ = torch.tensor([[0]], dtype=torch.float).to(opt.device)
                #d1_templ = torch.tensor([[1,1],[0,0]], dtype=torch.float).to(opt.device)
                #d2_templ = torch.tensor([[0,0],[1,1]], dtype=torch.float).to(opt.device)
                d1_scale = F.interpolate(d1_templ[None,None,...], size=(r,c), mode='nearest')
                d2_scale = F.interpolate(d2_templ[None,None,...], size=(r,c), mode='nearest')
            
                # Drop batch dimension but leave embedding dimension.  (If interpolate used.)
                d1_scale = d1_scale[0]
                d2_scale = d2_scale[0]

            errD1_real = -output1.mean()#-a
            errD1_real.backward(retain_graph=True)
            D1_x = -errD1_real.item()

            errD2_real = -output2.mean()#-a
            errD2_real.backward(retain_graph=True)
            D2_x = -errD2_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(g_real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs,Zs,g_reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs,Zs,g_reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(g_real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,g_reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev

            # Should this be one or two fakes?
            fake = netG(noise.detach(),prev)
            output1 = netD1(fake.detach())
            errD1_fake = output1.mean()
            errD1_fake.backward(retain_graph=True)
            D1_G_z = output1.mean().item()

            output2 = netD2(fake.detach())
            errD2_fake = output2.mean()
            errD2_fake.backward(retain_graph=True)
            D2_G_z = output2.mean().item()

            gradient_penalty_1 = functions.calc_gradient_penalty(netD1, d1_real, fake, opt.lambda_grad, opt.device)
            gradient_penalty_1.backward()

            gradient_penalty_2 = functions.calc_gradient_penalty(netD2, d2_real, fake, opt.lambda_grad, opt.device)
            gradient_penalty_2.backward()

            errD1 = errD1_real + errD1_fake + gradient_penalty_1
            errD2 = errD2_real + errD2_fake + gradient_penalty_2
            optimizerD1.step()
            optimizerD2.step()

        errD1plot.append(errD1.detach())
        errD2plot.append(errD2.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            fake = netG(noise.detach(),prev.detach())

            # No scaling
            #output1 = netD1(fake)
            #output2 = netD2(fake)

            #errG1 = -output1.mean()
            #errG1.backward(retain_graph=True)

            #errG2 = -output2.mean()
            #errG2.backward(retain_graph=True)

            # Scaling
            output1 = netD1(fake) * d1_scale * -1.
            output2 = netD2(fake) * d2_scale * -1.

            errG1 = output1.mean()
            output1.backward(gradient=torch.ones_like(output1), retain_graph=True)

            errG2 = output2.mean()
            output2.backward(gradient=torch.ones_like(output2), retain_graph=True)

            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_prev
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev),g_real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        errGplot.append(errG1.detach()+errG2.detach()+rec_loss)
        D1_real2plot.append(D1_x)
        D1_fake2plot.append(D1_G_z)
        D2_real2plot.append(D2_x)
        D2_fake2plot.append(D2_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d/%d: [%d/%d]' % (len(Gs), opt.stop_scale, epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD1.step()
        schedulerD2.step()
        schedulerG.step()

    functions.save_networks(netG,netD1,netD2,z_opt,opt)
    return z_opt,in_s,netG    

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD1 = models.WDiscriminator(opt).to(opt.device)
    netD1.apply(models.weights_init)
    if opt.netD1 != '':
        netD1.load_state_dict(torch.load(opt.netD))
    print(netD1)

    netD2 = models.WDiscriminator(opt).to(opt.device)
    netD2.apply(models.weights_init)
    if opt.netD2 != '':
        netD2.load_state_dict(torch.load(opt.netD2))
    print(netD2)

    return netD1, netD2, netG
