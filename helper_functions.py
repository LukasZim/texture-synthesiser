import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from helper_functions import *

# loads image from path
def image_loader(path, img_size=128, device='cpu'):
    img = Image.open(path)
    img = transforms.Resize((img_size, img_size))(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img.to(device, torch.float)

# saves the image to out_folder
def save_image(tensor, title=None, out_folder=None):
    assert out_folder is not None and title is not None
    plt.figure()
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)

    fig_path = os.path.join(out_folder, f'{title}.png')
    plt.savefig(fig_path, bbox_inches='tight')

    plt.close()

# the network
class Vgg19(torch.nn.Module):
    def __init__(self, content_layers, style_layers, device):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()

        self.slices = []
        self.layer_names = []
        self._remaining_layers = set(content_layers + style_layers)
        self._conv_names = [
            'conv1_1', 'conv1_2',
            'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
        ]

        i = 0
        model = torch.nn.Sequential()

        # add all layers we want to the network
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                name = self._conv_names[i]
                i += 1

            elif isinstance(layer, nn.ReLU):
                name = 'relu{}'.format(i)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool{}'.format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn{}'.format(i)

            model.add_module(name, layer)
            self.slices.append(model)

            self.layer_names.append(name)
            model = torch.nn.Sequential()

    # the forward pass of the network
    def forward(self, x):
        outs = []
        for slice in self.slices:
            x = slice(x)
            outs.append(x.clone())

        out = dict(zip(self.layer_names, outs))
        return out
