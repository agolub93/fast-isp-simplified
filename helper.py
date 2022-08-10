import torch, torch.nn as nn
from torchvision import models

# helper functions for model training.

# batch normalization
def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

# vgg-19 network for content loss.
def vgg_19(device):
    content_layer = 'relu_16'
    vgg_19 = models.vgg19(pretrained = True).features
    model = nn.Sequential()
    i = 0

    for layer in vgg_19.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name == content_layer:
            break

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in vgg_19.parameters():
        param.requires_grad = False

    return model

