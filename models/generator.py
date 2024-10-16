# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock


class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """ Patch based Generator. Uses namespace opt. """
    def __init__(self, opt, real):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.no_softmax = opt.no_softmax
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_current, N, (opt.g_ker_size, opt.g_ker_size), opt.g_padding, opt.g_padding_mode, opt.g_stride)  # Padding is done externally
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, (opt.g_ker_size, opt.g_ker_size), opt.g_padding, opt.g_padding_mode, opt.g_stride)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, (opt.g_ker_size, opt.g_ker_size), opt.g_padding, opt.g_padding_mode, opt.g_stride)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        self.tail = nn.Sequential(nn.Conv2d(N, opt.nc_current, kernel_size=(opt.g_ker_size, opt.g_ker_size),
                                            stride=opt.g_stride, padding=opt.g_padding, padding_mode=opt.g_padding_mode))
        
        #nodes = opt.nc_current * real.shape[-2] * real.shape[-1]
        #self.fc = nn.Linear(nodes, nodes)

    def forward(self, x, y, temperature=1):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        #x_shape = x.shape
        #x = self.fc(x.flatten())
        #x = x.reshape(x_shape)

        if not self.no_softmax:
            x = F.softmax(x * temperature, dim=1)  # Softmax is added here to allow for the temperature parameter
        
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

