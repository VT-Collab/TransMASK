from typing import OrderedDict
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from sparsemax import Sparsemax
import numpy as np
import math
from torchvision import transforms as T
from byol_pytorch import BYOL

from utils import *


class ImageEncoder(nn.Module):
    def __init__(self, method_params, feature_dim=128):
        super(ImageEncoder, self).__init__()

        weights = ResNet18_Weights.DEFAULT
        cnn_backbone = resnet18(weights)

        self._replace_bn_with_gn(cnn_backbone)
        cnn_backbone.fc = nn.Identity()
        enc_list = [cnn_backbone]

        linear = nn.Linear(512, feature_dim)
        enc_list.append(linear)
        self.enc = nn.Sequential(*enc_list)

    def forward(self, x):
        features = self.enc(x)
        return features
    
    def _replace_bn_with_gn(self, cnn, num_groups=32):
        for name, child in cnn.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features

                g = min(num_groups, num_channels)
                while num_channels % g != 0:
                    g -= 1

                setattr(cnn, name, nn.GroupNorm(num_groups=g,
                                                num_channels=num_channels,
                                                eps=child.eps,
                                                affine=True))
            else:
                self._replace_bn_with_gn(child, num_groups)


class VAEEncoder(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dims):
        super(VAEEncoder, self).__init__()

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

    def reparameterize(self, mu, logvar):
        z = mu + torch.exp(0.5 * logvar)
        return z
    
    def forward(self, s):
        z = self.encoder(s)
        z_mean = self.mean_layer(z)
        z_logvar = self.logvar_layer(z)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dims):
        super(VAEDecoder, self).__init__()

        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        
        self.pi = nn.Sequential(*layers)

    def forward(self, z):
        h = self.pi(z)
        return torch.tanh(h)
    

class Encoder(nn.Module):
    def __init__(self, state_dim, method_params, latent_dim=128, hidden_dims=[256, 256, 256], activation='softmax', sequential=False, use_images=False):
        super(Encoder, self).__init__()

        self.method_params = method_params
        self.x_dim = state_dim
        self.activation = activation
        self.sequential = sequential
        self.use_images = use_images
        
        if use_images:
            self.visual_encoders = nn.ModuleDict()
            self.visual_encoders['static_image'] = ImageEncoder(method_params=method_params)
            self.visual_encoders['ee_image'] = ImageEncoder(method_params=method_params)
        if method_params['use_mask']:
            layers = []
            layers.append(nn.Linear(state_dim, hidden_dims[0]))
            layers.append(nn.ReLU())
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[-1], (state_dim) ** 2))
            self.encoder = nn.Sequential(*layers)
        if method_params['use_vae']:
            self.vae_encoder = VAEEncoder(state_dim=state_dim,
                                          latent_dim=latent_dim,
                                          hidden_dims=hidden_dims)

        if self.activation == 'sparsemax':
            self.sparsemax = Sparsemax(dim=-1)

    def forward(self, batch):
        if self.use_images:
            image_keys = [key for key in batch.keys() if 'image' in key]

            h = []
            for key in image_keys:
                img = batch[key]
                shape = img.shape[:-3]
                if len(shape) > 1:
                    img = img.flatten(end_dim=-4)
                x = self.visual_encoders[key](img)
                h.append(x.reshape(*shape, -1))
            h = torch.cat(h, dim=-1)
            features = torch.cat((h, batch['observation'], batch['feature']), dim=-1)
        else:
            features = batch['observation']

        if not np.any([v for _, v in self.method_params.items()]):
            return features, None, None
        
        elif self.method_params['use_mask']:
            static_vector = torch.ones_like(features)
            M_ = self.encoder(static_vector)
            if not self.sequential:
                M_ = M_.reshape(-1, self.x_dim, self.x_dim)
            else:
                B, T, N = features.shape
                M_ = M_.reshape(-1, T, self.x_dim, self.x_dim)
            
            if self.activation == 'softmax':
                M = torch.softmax(M_, dim=-1)
            elif self.activation == 'sparsemax':
                M = self.sparsemax(M_)
            else:
                M = M_

            if not self.sequential:
                return torch.bmm(M, features[:, :, None]).squeeze(-1), M_, M
            else:
                return torch.bmm(M.reshape(B * T, N, N), features.reshape(B * T, N, 1)).reshape(B, T, N), M_, M
        
        elif self.method_params['use_vae']:
            z, z_mean, z_logvar = self.vae_encoder(features)
            return z, z_mean, z_logvar
