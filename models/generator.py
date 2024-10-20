# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock

def add_coords(x):
    batch, chan, height, width = x.shape

    # Create normalized coordinate grids
    y_coords = torch.linspace(-1, 1, height).reshape(1, 1, height, 1).expand(batch, 1, height, width)
    x_coords = torch.linspace(-1, 1, width).reshape(1, 1, 1, width).expand(batch, 1, height, width)

    # Concatenate to existing features
    return torch.cat([x, y_coords.to(x.device), x_coords.to(x.device)], dim=1)


class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """ Patch based Generator. Uses namespace opt. """
    def __init__(self, opt):
        super().__init__()
        if opt.repr_type == 'one-hot': self.use_softmax = True
        else: self.use_softmax = False
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_current, N, (opt.ker_size, opt.ker_size), 0, 1)  # Padding is done externally
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        self.tail = nn.Sequential(nn.Conv2d(N, opt.nc_current, kernel_size=(opt.ker_size, opt.ker_size),
                                            stride=1, padding=0))

        print(opt.nc_current, opt.nzx_current, opt.nzy_current)
        self.n_out = opt.nc_current * opt.nzx_current * opt.nzy_current
        print("making linear of", self.n_out,"x", self.n_out, "(", self.n_out * self.n_out, "parameters)")
        self.fc = nn.Linear(self.n_out, self.n_out).to(opt.device)
        self.coord = nn.Conv2d(opt.nc_current+2, opt.nc_current, 1)

    def forward(self, x, y, temperature=1):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        # If FC or coords.
        x_shape = x.shape

        # If coords.
        #x = add_coords(x)
        #x = self.coord(x)

        # If FC.
        x = self.fc(x.flatten(start_dim=1))

        # If FC or coords.
        x = x.reshape(x_shape)

        if self.use_softmax: x = F.softmax(x * temperature, dim=1)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

