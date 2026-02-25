import os
from cv2 import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from models import *
from utils import *


class MLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dims, method_params, activation='softmax', using_gripper=False, use_images=False):
        super(MLPPolicy, self).__init__()

        self.method_params = method_params
        self.using_gripper = using_gripper
        self.use_images = use_images

        self.encoder = Encoder(state_dim=state_dim,
                               latent_dim=latent_dim,
                               hidden_dims=hidden_dims,
                               method_params=method_params,
                               activation=activation,
                               use_images=use_images)

        layers = []
        if method_params['use_vae']:
            layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        else:
            layers.append(nn.Linear(state_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        self.pi = nn.Sequential(*layers)

    def forward(self, batch):
        z, M_, M = self.encoder(batch)

        a = self.pi(z)
        if not self.using_gripper:
            a = torch.tanh(a)
            loss = torch.nn.MSELoss()(a, batch['action'])
        else:
            a[:, -1] = torch.sigmoid(a[:, -1])
            a[:, :-1] = torch.tanh(a[:, :-1])
            batch['action'][batch['action'][:, -1] == -1, -1] = 0
            loss = torch.nn.MSELoss()(a[:, :-1], batch['action'][:, :-1]) + torch.nn.functional.binary_cross_entropy(a[:, -1], batch['action'][:, -1])
        return z, M_, M, loss
    
    def get_action(self, obs, foo=None):
        z, M_, M = self.encoder(obs)
        a = self.pi(z)
        if not self.using_gripper:
            a = torch.tanh(a)
        else:
            a[:, -1] = torch.sigmoid(a[:, -1])
            a[:, :-1] = torch.tanh(a[:, :-1])
            a[a[:, -1] < 0.7, -1] = -1
            a[a[:, -1] >= 0.7, -1] = 1
        a = a.detach().cpu().numpy()
        z = z.detach().cpu().numpy().squeeze()
        if M is not None:
            M_ = M_.detach().cpu().numpy().squeeze()
            M = M.detach().cpu().numpy().squeeze()
        return a, z, M_, M

        
class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, emb_dim, hidden_dims,
                 n_heads, n_layers, timesteps, obs_horizon, pred_horizon, n_rollout_actions,
                 method_params, activation, sequential, device, use_images=False):
        super(DiffusionPolicy, self).__init__()

        self.method_params = method_params
        self.gamma_scheduler = DDIMScheduler()
        self.device = device
        self.timesteps = timesteps
        self.seq_in = obs_horizon
        self.seq_out = pred_horizon
        self.activation = activation
        self.n_rollout_actions = n_rollout_actions

        self.encoder = Encoder(state_dim=state_dim,
                               latent_dim=latent_dim,
                               hidden_dims=hidden_dims,
                               method_params=method_params,
                               activation=activation,
                               sequential=sequential,
                               use_images=use_images)

        if method_params['use_vae']:
            self.decoder = VAEDecoder(latent_dim=latent_dim,
                                      action_dim=action_dim,
                                      hidden_dims=hidden_dims)
            
        self.denoiser = ActionSequenceDenoiser(state_dim=latent_dim if method_params['use_vae'] else state_dim,
                                               action_dim=action_dim,
                                               emb_dim=emb_dim,
                                               n_heads=n_heads,
                                               n_layers=n_layers,
                                               mlp_hidden=hidden_dims[0],
                                               obs_horizon=obs_horizon,
                                               pred_horizon=pred_horizon)
        
        self.gamma_scheduler.set_timesteps(timesteps)

    def q_sample(self, action, t, noise=None):
        if noise is None:
            noise = torch.randn_like(action)
        return self.gamma_scheduler.add_noise(action, noise, t)

    def forward(self, batch):
        action_seq = batch['action']
        B = action_seq.shape[0]
        t = torch.randint(0, self.gamma_scheduler.config.num_train_timesteps, (B,), device=self.device)
        noise = torch.randn_like(action_seq)
        noisy_action = self.q_sample(action_seq, t, noise)

        z, M_, M = self.encoder(batch)

        pred_noise = self.denoiser(noisy_action, z, t)
        loss = F.mse_loss(pred_noise, noise)

        if self.method_params['use_vae']:
            a_recon = self.decoder(z)
            loss += F.mse_loss(a_recon, batch['action'][:, :self.seq_in, :])
        return z, M_, M, loss

    @torch.no_grad()
    def get_action(self, obs, inference_timesteps=None):
        B = obs['observation'].shape[0]
        z, M_, M = self.encoder(obs)

        action = torch.randn(B, self.seq_out, self.denoiser.output_mlp[-1].out_features, device=self.device)

        if inference_timesteps is None:
            inference_timesteps = self.timesteps

        scheduler = self.gamma_scheduler
        scheduler.set_timesteps(inference_timesteps)

        for t in scheduler.timesteps:
            t_batch = torch.full((B,), t, device=self.device)
            eps_pred = self.denoiser(action, z, t_batch)
            action = scheduler.step(eps_pred, t, action).prev_sample

        a = action.detach().cpu().numpy().squeeze()
        z = z.detach().cpu().numpy().squeeze()
        if M is not None:
            M_ = M_.detach().cpu().numpy().squeeze()
            M = M.detach().cpu().numpy().squeeze()
        return a[self.seq_in:self.seq_in + self.n_rollout_actions, :], z, M_, M
    

class EMA:
    def __init__(self, ema_model, update_after_step=0, inv_gamma=1., power=0.75, min_value=0., max_value=0.999):
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.ema_model = ema_model
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

        self.decay = 0.
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))
    
    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.ema_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)
        
        self.optimization_step += 1
