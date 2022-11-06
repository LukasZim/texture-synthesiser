import torch
import torch.optim as optim

from helper_functions import *
out_folder = 'outputs'

# normalizes the image
def normalize(img, mean, std):
    """ Normalizes an image tensor.

    # Parameters:
        @img, torch.tensor of size (b, c, h, w)
        @mean, torch.tensor of size (c)
        @std, torch.tensor of size (c)

    # Returns the normalized image
    """
    # TODO: 1. Implement normalization doing channel-wise z-score normalization.
    # print("============================================")

    mean2 = mean.reshape(1,3,1,1)
    std2 = std.reshape(1,3,1,1)
    i2 = img - mean2
    i3 = i2 / std2
    # print(mean)
    # print(torch.sum(i3[0, 0]))

    return i3


def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.
    
    # Parameters:
        @x, torch.tensor of size (b, c, h, w) 

    # Returns the gram matrix
    """
    # Implements the calculation of the normalized gram matrix.
    b, c, h, w = x.size()

    features = x.view(b*c, h*w)

    G = torch.mm(features, features.t())
    H = G.div(c*h*w)
    return H

# calculates the style loss for 2 gram matrices
def calc_style_loss(G, A):
    return torch.mean(torch.square(G - A)) * 100000 # multiply because values are too small

# calculates the style loss, like in assignment 3
def style_loss(input_features, style_features, style_layers):
    sum = 0
    for layer in style_layers:
        input_feature_tensor = input_features[layer].requires_grad_(True)
        content_feature_tensor = style_features[layer].detach().requires_grad_(True)
        G = gram_matrix(input_feature_tensor)
        A = gram_matrix(content_feature_tensor)
        sum += calc_style_loss(G, A)
    return sum / len(style_layers)





