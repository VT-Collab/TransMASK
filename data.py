import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from aeon.distances import dtw_pairwise_distance
from tqdm import tqdm
from utils import *
from torchvision import transforms as T

import matplotlib.pyplot as plt


class Data(Dataset):
    def __init__(self, hdf5_path, use_images=False):
        super().__init__()

        self.hdf5_path = str(hdf5_path)
        self.use_images = use_images

        with h5py.File(self.hdf5_path, 'r') as f:
            self.demo_keys = sorted([k for k in f.keys() if 'demo' in k], key=lambda x: int(x.split('_')[1]))
            self.demo_lens = [f[k]['actions'].shape[0] for k in self.demo_keys]
            self.num_demos = len(self.demo_keys)

            self.action_shape = f[self.demo_keys[0]]['actions'].shape
            if 'feature' in f[self.demo_keys[0]]['obs'].keys():
                self.feature_shape = f[self.demo_keys[0]]['obs']['feature'].shape
            else:
                self.feature_shape = [0, 0, 0]

        self.cumulative_demo_lens = np.cumsum(self.demo_lens)

    def __len__(self):
        return int(self.cumulative_demo_lens[-1])

    def _get_hdf5(self):
        if not hasattr(self, '_hdf5'):
            self._hdf5 = h5py.File(self.hdf5_path, 'r')
        return self._hdf5

    def __getitem__(self, index):
        demo_idx = np.searchsorted(self.cumulative_demo_lens, index, side='right')
        local_idx = index if demo_idx == 0 else index - self.cumulative_demo_lens[demo_idx - 1]

        f = self._get_hdf5()
        demo = f[self.demo_keys[demo_idx]]

        if not self.use_images:
            obs = demo['obs']['observation'][local_idx]
            goal = demo['obs']['desired_goal'][local_idx]
            obs = np.concatenate((obs, goal), axis=-1)
            actions = demo['actions'][local_idx]
            static = ego = features = None
        else:
            actions = demo['actions'][local_idx]
            obs = demo['obs']['observation'][local_idx, :actions.shape[-1]]
            static = demo['obs']['static_image'][local_idx]
            ego = demo['obs']['ee_image'][local_idx]
            features = np.concatenate((demo['obs']['static_seg'][local_idx],
                                       demo['obs']['ee_seg'][local_idx]),
                                       axis=-1)

        out = {'observation': torch.tensor(obs, dtype=torch.float32),
               'action': torch.tensor(actions, dtype=torch.float32)}

        if self.use_images:
            out.update({'static_image': torch.tensor(static, dtype=torch.float32).permute(2, 0, 1) / 255.0,
                        'ee_image': torch.tensor(ego, dtype=torch.float32).permute(2, 0, 1) / 255.0,
                        'feature': torch.tensor(features, dtype=torch.float32)})
        return out


class DataSequence(Dataset):
    def __init__(self, hdf5_path, obs_horizon=8, pred_horizon=4, use_images=False):
        super().__init__()

        self.hdf5_path = str(hdf5_path)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_images = use_images

        with h5py.File(self.hdf5_path, 'r') as f:
            self.demo_keys = sorted([k for k in f.keys() if 'demo' in k], key=lambda x: int(x.split('_')[1]))
            self.demo_lens = [f[k]['actions'].shape[0] for k in self.demo_keys]
            self.num_demos = len(self.demo_keys)

            self.action_shape = f[self.demo_keys[0]]['actions'].shape
            if 'feature' in f[self.demo_keys[0]]['obs'].keys():
                self.feature_shape = f[self.demo_keys[0]]['obs']['feature'].shape
            else:
                self.feature_shape = [0, 0, 0]

        self.cumulative_demo_lens = np.cumsum(self.demo_lens)

    def __len__(self):
        return int(self.cumulative_demo_lens[-1])

    def _get_hdf5(self):
        if not hasattr(self, '_hdf5'):
            self._hdf5 = h5py.File(self.hdf5_path, 'r')
        return self._hdf5

    def __getitem__(self, index):
        demo_idx = np.searchsorted(self.cumulative_demo_lens, index, side='right')
        local_idx = index if demo_idx == 0 else index - self.cumulative_demo_lens[demo_idx - 1]

        f = self._get_hdf5()
        demo = f[self.demo_keys[demo_idx]]
        demo_len = self.demo_lens[demo_idx]

        obs_end = min(local_idx + self.obs_horizon, demo_len)
        act_end = min(local_idx + self.pred_horizon, demo_len)

        if not self.use_images:
            obs = demo['obs']['observation'][local_idx:obs_end]
            goal = demo['obs']['desired_goal'][local_idx:obs_end]
            obs = np.concatenate((obs, goal), axis=-1)
            actions = demo['actions'][local_idx:act_end]
            static = ego = feat = None
        else:
            actions = demo['actions'][local_idx:act_end]
            obs = demo['obs']['observation'][local_idx:obs_end, :actions.shape[-1]]
            static = demo['obs']['static_image'][local_idx:obs_end]
            ego = demo['obs']['ee_image'][local_idx:obs_end]
            feat = np.concatenate((demo['obs']['static_seg'][local_idx:obs_end],
                                   demo['obs']['ee_seg'][local_idx:obs_end]),
                                   axis=-1)

        if obs.shape[0] < self.obs_horizon:
            pad = self.obs_horizon - obs.shape[0]
            obs = np.concatenate([obs, np.repeat(obs[-1:], pad, axis=0)])
            if self.use_images:
                static = np.concatenate([static, np.repeat(static[-1:], pad, axis=0)])
                ego = np.concatenate([ego, np.repeat(ego[-1:], pad, axis=0)])
                feat = np.concatenate([feat, np.repeat(feat[-1:], pad, axis=0)])

        if actions.shape[0] < self.pred_horizon:
            pad = self.pred_horizon - actions.shape[0]
            actions = np.concatenate([actions, np.repeat(actions[-1:], pad, axis=0)])

        out = {'observation': torch.tensor(obs, dtype=torch.float32),
               'action': torch.tensor(actions, dtype=torch.float32)}

        if self.use_images:
            out.update({'static_image': torch.tensor(static, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0,
                        'ee_image': torch.tensor(ego, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0,
                        'feature': torch.tensor(feat, dtype=torch.float32)})
        return out

