import torch
import torch.optim as optim

from helper_functions import *
out_folder = 'outputs'
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

def calc_content_loss(input_feature_tensor, content_feature_tensor):
    return torch.mean(torch.square(input_feature_tensor - content_feature_tensor))

# def content_loss(input_features, content_features, content_layers):
#     """ Calculates the content loss as in Gatys et al. 2016.
#
#     # Parameters:
#         @input_features, VGG features of the image to be optimized. It is a
#             dictionary containing the layer names as keys and the corresponding
#             features volumes as values.
#         @content_features, VGG features of the content image. It is a dictionary
#             containing the layer names as keys and the corresponding features
#             volumes as values.
#         @content_layers, a list containing which layers to consider for calculating
#             the content loss.
#
#     # Returns the content loss, a torch.tensor of size (1)
#     """
#     # TODO: 2. Implement the content loss given the input feature volume and the
#     # content feature volume. Note that:
#     # - Only the layers given in content_layers should be used for calculating this loss.
#     # - Normalize the loss by the number of layers.
#
#     sumTensor = torch.tensor(0.0, requires_grad=True)
#     for layer in content_layers:
#         input_feature_tensor = input_features[layer].detach()
#
#         content_feature_tensor = content_features[layer].detach()
#
#
#         loss = calc_content_loss(input_feature_tensor, content_feature_tensor)
#         sumTensor = torch.add(sumTensor, loss)
#     sumTensor.detach()
#     answer = sumTensor / torch.tensor(len(content_layers))
#
#
#     # total_loss= torch.where(content_layers, torch.mean(torch.square(input_features[layer] - content_features[layer])), 0)
#     # print(answer)
#     return answer

def content_loss(input_features, content_features, content_layers):
    sumTensor = 0
    for layer in content_layers:
        input_feature_tensor = input_features[layer].requires_grad_(True)

        content_feature_tensor = content_features[layer].detach().requires_grad_(True)

        # loss =
        sumTensor +=  calc_content_loss(input_feature_tensor, content_feature_tensor)/(len(content_layers))
    # answer = sumTensor
    return sumTensor

def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.
    
    # Parameters:
        @x, torch.tensor of size (b, c, h, w) 

    # Returns the gram matrix
    """
    # TODO: 3.2 Implement the calculation of the normalized gram matrix. 
    # Do not use for-loops, make use of Pytorch functionalities.
    b, c, h, w = x.size()

    features = x.view(b*c, h*w)

    G = torch.mm(features, features.t())
    H = G.div(c*h*w)
    return H

def calc_style_loss(G, A):
    return torch.mean(torch.abs(G - A))

def style_loss(input_features, style_features, style_layers):
    sum = 0
    for layer in style_layers:
        input_feature_tensor = input_features[layer].requires_grad_(True)
        content_feature_tensor = style_features[layer].detach().requires_grad_(True)
        G = gram_matrix(input_feature_tensor)
        A = gram_matrix(content_feature_tensor)
        w,x,y,z = content_feature_tensor.shape
        sum += calc_style_loss(A, G) #* (1/(4 * x**2 * (y*z)**2))
    if sum > 0.2:
        xd = sum
    return sum# / len(style_layers)

    # layer = style_layers[0]
    # total = calc_style_loss(gram_matrix(input_features[layer].detach()), gram_matrix(style_features[layer].detach()))
    #
    # for layer in style_layers[1:]:
    #     total = torch.add(total, calc_style_loss(gram_matrix(input_features[layer].detach()), gram_matrix(style_features[layer].detach())))
    # return torch.div(total, torch.tensor(len(style_layers)))

# def style_loss(input_features, style_features, style_layers):
#     """ Calculates the style loss as in Gatys et al. 2016.
#
#     # Parameters:
#         @input_features, VGG features of the image to be optimized. It is a
#             dictionary containing the layer names as keys and the corresponding
#             features volumes as values.
#         @style_features, VGG features of the style image. It is a dictionary
#             containing the layer names as keys and the corresponding features
#             volumes as values.
#         @style_layers, a list containing which layers to consider for calculating
#             the style loss.
#
#     # Returns the style loss, a torch.tensor of size (1)
#     """
#     # TODO: 3.1 Implement the style loss given the input feature volume and the
#     # style feature volume. Note that:
#     # - Only the layers given in style_layers should be used for calculating this loss.
#     # - Normalize the loss by the number of layers.
#     # - Implement the gram_matrix function.
#
#     sumTensor = torch.tensor(0.0, requires_grad=True)
#     for x in input_features:
#         input_features[x] = input_features[x].detach()
#     for x in style_features:
#         style_features[x] = style_features[x].detach()
#     for layer in style_layers:
#         input_feature_tensor = input_features[layer].requires_grad_()
#         content_feature_tensor = style_features[layer].requires_grad_()
#
#         G = gram_matrix(input_feature_tensor)
#         A = gram_matrix(content_feature_tensor)
#         b, c, h, w = input_feature_tensor.size()
#         loss = calc_style_loss(G, A)
#         sumTensor = torch.add(sumTensor, loss)
#     answer = torch.div(sumTensor, torch.tensor(len(style_layers)))
#     return answer
#
#     # total_loss = [0] * len(style_layers)
#     # index = 0
#     # for layer in style_layers:
#     #     input_feature_tensor = input_features[layer]
#     #     content_feature_tensor = style_features[layer]
#     #     bs1, N, M, bs2 = input_feature_tensor.size()
#     #
#     #
#     #     loss = torch.sum(torch.square(content_feature_tensor - input_feature_tensor))
#     #     total_loss[index] = (1/(4*pow(N,2)*pow(M,2)))*loss
#     #
#     #     index += 1
#     # answer = torch.sum(torch.tensor(total_loss)) / len(style_layers)
#     # return answer

def total_variation_loss(input):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)
    
    # Returns the total variation, a torch.tensor of size (1)
    """
    # TODO: 4. Implement the total variation loss.

    ans = torch.abs(torch.diff(input, dim=2)[:,:,:127,:127])
    ans = torch.add(ans, torch.abs(torch.diff(input, dim=3)[:,:,:127,:127]))
    ans = torch.sum(ans)
    return ans


