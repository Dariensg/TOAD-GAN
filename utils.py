import random

import numpy as np
import torch
import torchvision
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import math


def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def prepare_mnist_seed_images():
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('output/mnist/', train=False, download=True,
                                   transform=torchvision.transforms.Compose(
                                       [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=1, shuffle=True)
    eights = torch.zeros((20, 1, 28, 28))
    e = 0
    while e < eights.shape[0]:
        batch = next(iter(test_loader))
        if batch[1].item() == 8:
            eights[e] = batch[0]
            e += 1

    for i in range(len(eights)):
        tmp = eights[i, 0]
        x, y = torch.where(tmp > 0)
        l_x = max(x) - min(x)
        l_y = max(y) - min(y)
        if l_x == l_y:
            x_1 = min(x)
            x_2 = max(x) + 2
            y_1 = min(y)
            y_2 = max(y) + 2
        elif l_x > l_y:
            x_1 = min(x)
            x_2 = max(x) + 2
            diff = l_x - l_y
            y_1 = min(y) - diff//2
            y_2 = max(y) + diff//2 + 2
        else:  # l_y > l_x:
            y_1 = min(y)
            y_2 = max(y) + 2
            diff = l_y - l_x
            x_1 = min(x) - diff//2
            x_2 = max(x) + diff//2 + 2
        tmp = tmp[x_1:x_2, y_1:y_2]
        # tmp = interpolate(tmp.unsqueeze(0).unsqueeze(0), (28, 28))
        plt.imsave('mariokart/seed_road/MNIST_examples/eights/sample_%d.png' % i, tmp[0][0], cmap='Greys')

def get_discriminator1_scaling_tensor(opt, outputD1):
    if (opt.alpha_layer_type == "half-and-half"):
        zeros = torch.tensor([[[0.] * math.ceil(outputD1.size()[-1] / 2)] * outputD1.size()[-2]])
        ones = torch.tensor([[[1.] * math.floor(outputD1.size()[-1] / 2)] * outputD1.size()[-2]])
        return torch.cat((zeros, ones), 2).to(opt.device)
    elif (opt.alpha_layer_type == "all-ones"):
        return torch.ones_like(outputD1).to(opt.device)
    elif (opt.alpha_layer_type == "all-zeros"):
        return torch.zeros_like(outputD1).to(opt.device)

def get_discriminator2_scaling_tensor(opt, outputD2):
    if (opt.alpha_layer_type == "half-and-half"):
        zeros = torch.tensor([[[0.] * math.ceil(outputD2.size()[-1] / 2)] * outputD2.size()[-2]])
        ones = torch.tensor([[[1.] * math.floor(outputD2.size()[-1] / 2)] * outputD2.size()[-2]])
        return torch.cat((ones, zeros), 2).to(opt.device)
    elif (opt.alpha_layer_type == "all-ones"):
        return torch.zeros_like(outputD2).to(opt.device)
    elif (opt.alpha_layer_type == "all-zeros"):
        return torch.ones_like(outputD2).to(opt.device)