import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import copy
import torch.distributed as dist
from collections import OrderedDict


def get_device(device):
    if device == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available() and device == 'cuda':
        return torch.device('cuda')
    if torch.backends.mps.is_available() and device == 'mps':
        return torch.device('mps')
    raise ValueError('Invalid device name')


class ObservationBuffer:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.seq_len = args.obs_horizon if args.sequential else 1
        self.buffer = []

    def reset(self):
        self.buffer.clear()

    def _copy_obs(self, obs):
        obs_copy = {}
        for k, v in obs.items():
            if isinstance(v, dict):
                obs_copy[k] = {kk: np.copy(vv) for kk, vv in v.items()}
            else:
                obs_copy[k] = np.copy(v)
        return obs_copy

    def add(self, obs):
        obs_copy = self._copy_obs(obs)

        if self.args.return_image_obs:
            obs_copy['observation'] = obs_copy['observation'][:self.env.action_space.shape[0]]
            obs_copy['feature'] = np.concatenate((obs_copy['static_seg'], obs_copy['ee_seg']), axis=-1)
            obs_copy['static_image'] = np.transpose(obs_copy['static_image'], (2, 0, 1)) / 255.
            obs_copy['ee_image'] = np.transpose(obs_copy['ee_image'], (2, 0, 1)) / 255.
        else:
            obs_copy['observation'] = np.concatenate((obs_copy['observation'], obs_copy['desired_goal']))
            
        self.buffer.append(obs_copy)

        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        while len(self.buffer) < self.seq_len:
            self.buffer.insert(0, copy.deepcopy(self.buffer[0]))

    def get_sequence(self):
        if not self.buffer:
            raise RuntimeError("Observation buffer is empty.")
        seq_obs = {}

        keys = self.buffer[0].keys()
        for k in keys:
            seq_obs[k] = np.stack([step[k] for step in self.buffer], axis=0)
            if not self.args.sequential:
                seq_obs[k] = seq_obs[k].squeeze()
        return seq_obs


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb
    

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, positions):
        return self.pe[positions]
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_hidden):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden),
                                 nn.SiLU(),
                                 nn.Linear(mlp_hidden, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x_action, x_state, tgt_mask, memory_mask):
        h = self.self_attn(self.norm1(x_action), self.norm1(x_action), self.norm1(x_action), attn_mask=tgt_mask)[0] + x_action
        
        h = self.cross_attn(self.norm2(h), self.norm2(x_state), self.norm2(x_state), attn_mask=memory_mask)[0] + h
        
        h = self.mlp(self.norm3(h)) + h
        return h
    

class ActionSequenceDenoiser(nn.Module):
    def __init__(self, state_dim, action_dim, emb_dim, n_heads, n_layers, mlp_hidden, obs_horizon, pred_horizon, max_seq_len=1000):
        super(ActionSequenceDenoiser, self).__init__()

        self.state_proj = nn.Linear(state_dim, emb_dim)
        self.action_proj = nn.Linear(action_dim, emb_dim)
        self.t_emb = SinusoidalPositionEmbeddings(emb_dim)
        self.pos_encoding = PositionalEncoding(emb_dim, max_len=max_seq_len)

        self.layers = nn.ModuleList([CrossAttentionBlock(emb_dim, n_heads, mlp_hidden) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(emb_dim)
        
        self.output_mlp = nn.Sequential(nn.Linear(emb_dim, mlp_hidden),
                                        nn.SiLU(),
                                        nn.Linear(mlp_hidden, action_dim))
        
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, obs_horizon+1, emb_dim), requires_grad=True)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, pred_horizon+1, emb_dim), requires_grad=True)

        T = pred_horizon
        tgt_mask = (torch.triu(torch.ones(T, T)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        self.register_buffer("tgt_mask", tgt_mask)

        T_cond = obs_horizon + 1
        t, s = torch.meshgrid(torch.arange(T),
                              torch.arange(T_cond),
                              indexing='ij')
        memory_mask = t >= (s-1)
        memory_mask = memory_mask.float().masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1, float(0.0))
        self.register_buffer("memory_mask", memory_mask)

    def forward(self, noisy_action_seq, state_seq, timesteps):
        t_emb = self.t_emb(timesteps).unsqueeze(1)
        state_proj = self.state_proj(state_seq)
        cond_emb = torch.cat((t_emb, state_proj), dim=1)
        
        tc = cond_emb.shape[1]
        pos_emb_state = self.cond_pos_emb[:, :tc, :] 
        state_emb = cond_emb + pos_emb_state

        ta = noisy_action_seq.shape[1]
        pos_emb_action = self.action_pos_emb[:, :ta, :]
        action_emb = self.action_proj(noisy_action_seq) + pos_emb_action

        for layer in self.layers:
            action_emb = layer(action_emb, state_emb, self.tgt_mask, self.memory_mask)
        
        out_action = self.norm(action_emb)
        eps_pred = self.output_mlp(out_action)
        return eps_pred
    

