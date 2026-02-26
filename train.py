import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import json
import gymnasium as gym
import panda_gym
import copy

from policies import MLPPolicy, DiffusionPolicy, EMA
from utils import get_device
from data import Data, DataSequence
from config import get_config


def train(args):
    device = get_device(args.device)

    if not args.sequential:
        dataset = Data(hdf5_path=args.dataset_path,
                       use_images=args.return_image_obs)
    else:
        dataset = DataSequence(hdf5_path=args.dataset_path,
                               obs_horizon=args.obs_horizon,
                               pred_horizon=args.pred_horizon,
                               use_images=args.return_image_obs)
    
    if 'PickAndPlace' in args.env_name:
        task_name = 'PickAndPlace'
    elif 'Flip' in args.env_name:
        task_name = 'Flip'
    elif 'Push' in args.env_name:
        task_name = 'Push'
    else:
        raise ValueError("Undefined task")
    
    saveloc = os.path.join(os.getcwd(), args.saveloc, task_name, args.savename)
    os.makedirs(saveloc, exist_ok=True)

    env = gym.make(f"{args.env_name}", render_mode=args.render_mode, num_distractors=args.num_distractors, return_image_obs=args.return_image_obs)
    if not args.return_image_obs:
        obs_shape = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    else:
        obs_shape = env.action_space.shape[0] + 128 + 128 + env.observation_space['ee_seg'].shape[0] + env.observation_space['static_seg'].shape[0]

    arguments = vars(args)
    with open(os.path.join(saveloc, 'arguments.json'), "w") as f:
        json.dump(arguments, f, indent=4)

    if args.env_name in ['PandaPush-v3']:
        using_gripper = False
    else:
        using_gripper = True
    if args.policy == 'mlp':
        policy = MLPPolicy(state_dim=obs_shape,
                           action_dim=env.action_space.shape[0],
                           latent_dim=args.latent_dim,
                           hidden_dims=args.hidden_dims,
                           method_params=args.method_params,
                           activation=args.activation,
                           using_gripper=using_gripper,
                           use_images=args.return_image_obs)
    elif args.policy == 'diffusion':
        policy = DiffusionPolicy(state_dim=obs_shape,
                                 action_dim=env.action_space.shape[0],
                                 latent_dim=args.latent_dim,
                                 emb_dim=args.emb_dim,
                                 hidden_dims=args.hidden_dims,
                                 n_heads=args.n_heads,
                                 n_layers=args.n_layers,
                                 timesteps=args.timesteps,
                                 obs_horizon=args.obs_horizon,
                                 pred_horizon=args.pred_horizon,
                                 n_rollout_actions=args.n_rollout_actions,
                                 method_params=args.method_params,
                                 activation=args.activation,
                                 sequential=args.sequential,
                                 device=args.device,
                                 use_images=args.return_image_obs)
    else:
        raise ValueError(f"Unknown value passed for policy: {args.policy}")
    policy = policy.to(device)

    print(f'Total number of demos: {dataset.num_demos}')
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=args.batch_size)

    optimizer = Adam(policy.parameters(), lr=args.learning_rate)

    if args.use_ema_model:
        ema_policy = copy.deepcopy(policy).to(device)
        ema = EMA(ema_policy,
                  update_after_step=args.ema_update_after_step,
                  inv_gamma=args.ema_inv_gamma,
                  power=args.ema_power,
                  min_value=args.ema_min_value,
                  max_value=args.ema_max_value)
    
    print('='*25)
    print('Number of parameters: ', sum(p.numel() for p in policy.parameters()))
    print('='*25)
    
    LOSS = []
    if args.loadloc is not None:
        load_model = args.loadloc
        ckpt = torch.load(load_model, weights_only=True, map_location=device)
        policy.load_state_dict(ckpt['policy'])
        ema.ema_model.load_state_dict(ckpt['ema_policy'])
        optimizer.load_state_dict(ckpt['optimizer'])
        LOSS = ckpt['loss']

    start_epoch = len(LOSS)
    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs)):
        l = 0

        for i, batch in enumerate(dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            z, M_, M, loss = policy(batch)
            if args.method_params['use_vae']:
                loss += -0.5 * torch.sum(1 + M - M_.pow(2) - M.exp())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.use_ema_model:
                ema.step(policy)
            l += loss.item()

        LOSS.append(l)

        if epoch % 100 == 0:
            ckpt = {'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': LOSS}
            if args.use_ema_model:
                ckpt['ema_policy'] = ema.ema_model.state_dict()
            torch.save(ckpt, os.path.join(saveloc, f'ckpt_{epoch}.pt'))
    
    # Save final model
    ckpt = {'policy': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': LOSS}
    if args.use_ema_model:
        ckpt['ema_policy'] = ema.ema_model.state_dict()
    torch.save(ckpt, os.path.join(saveloc, 'best_model.pt'))


if __name__ == '__main__':
    args = get_config()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('='*25)
    print('Training with the following arguments')
    print(json.dumps(vars(args), indent=4))
    print('='*25)

    train(args)

    print('='*25)
    print('Done')
    print('='*25)