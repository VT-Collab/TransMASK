import os
import numpy as np
import torch
from tqdm import tqdm
import json
import gymnasium as gym
import panda_gym
import copy
import random

import matplotlib.pyplot as plt
import imageio

from policies import MLPPolicy, DiffusionPolicy, EMA
from utils import get_device, ObservationBuffer
from config import get_config


def evaluate(args):
    env = gym.make(f"{args.env_name}", render_mode=args.render_mode, domain_randomize=args.domain_randomize, num_distractors=args.num_distractors, return_image_obs=args.return_image_obs)
    if not args.return_image_obs:
        obs_shape = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    else:
        obs_shape = env.action_space.shape[0] + 128 + 128 + env.observation_space['ee_seg'].shape[0] + env.observation_space['static_seg'].shape[0]
    
    device = get_device(args.device)
    
    if args.policy == 'mlp':
        policy = MLPPolicy(state_dim=obs_shape,
                           action_dim=env.action_space.shape[0],
                           latent_dim=args.latent_dim,
                           hidden_dims=args.hidden_dims,
                           method_params=args.method_params,
                           activation=args.activation,
                           use_images=args.return_image_obs)
    elif args.policy == 'diffusion' and args.sequential:
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
    
    assert args.loadloc is not None, f"No pretrained model found at location: {args.loadloc}"
    load_model = os.path.join(args.loadloc, 'best_model.pt')

    print(f'Loading policy from {load_model}')
    ckpt = torch.load(load_model, weights_only=True, map_location=device)
    if args.use_ema_model:
        print('Using EMA policy for evaluation')
        ema_policy = copy.deepcopy(policy).to(device)
        state_dict = ckpt['ema_policy']
        ema_policy.load_state_dict(state_dict)
        eval_policy = ema_policy
    else:
        print('EMA not found, using standard policy')
        policy = policy.to(device)
        state_dict = ckpt['policy']
        policy.load_state_dict(state_dict)
        eval_policy = policy
    eval_policy.eval()

    training_loss = np.log([l.item() for l in ckpt['loss']])
    _, axs = plt.subplots(1, 1, figsize=(10, 5))

    axs.plot(training_loss)
    axs.set_title('Training Loss')

    plt.savefig(os.path.join(args.loadloc, 'loss.png'), dpi=300)
    plt.close()

    obs_buffer = ObservationBuffer(args, env)

    results = {}
    success_rate = 0
    masks = []
    if args.save_video:
        video_path = os.path.join(args.loadloc, 'results.mp4')
        video_writer = imageio.get_writer(video_path, fps=20)
        
    for eval in tqdm(range(args.num_evals), desc='Rolling out policy'):
        obs, _ = env.reset()
        obs_buffer.reset()
        obs_buffer.add(obs)
        reached_object = False
        
        frames = []
        for t in range(0, args.time_horizon, args.n_rollout_actions):
            obs_sequence = obs_buffer.get_sequence()
            for k, v in obs_sequence.items():
                obs_sequence[k] = torch.Tensor(v)[None, :].to(device)
            action, z, M_, M = eval_policy.get_action(obs_sequence, args.inference_steps)

            if args.method_params['use_mask'] and eval == 1:
                masks.append(M)
            
            for a_idx in range(args.n_rollout_actions):
                obs, _, _, _, _ = env.step(action[a_idx, :])
                obs_buffer.add(obs)
                
                success = env.unwrapped.task.is_success(obs['achieved_goal'], obs['desired_goal'])
                if args.save_video:
                    frames.append(env.render())
                if args.render_mode == 'human':
                    env.render()
                if success:
                    if len(frames) != 0:
                        for f in frames:
                            video_writer.append_data(f)
                    success_rate += 1
                    break
                if 'PickAndPlace' in args.env_name:
                    if t == args.time_horizon - 1 and obs['achieved_goal'][-1] > 3 * env.unwrapped.task.object_size:
                        success_rate += 0.5
                        break
                if args.domain_randomize:
                    if 'Push' in args.env_name:
                        if np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) < 0.1:
                            success_rate += 1
                            success = True
                            break
                    if 'Flip' in args.env_name:
                        if np.linalg.norm(env.unwrapped.robot.get_ee_position() - env.unwrapped.task.object_position) < 0.05 and not reached_object:
                            success_rate += 0.5
                            print('.')
                            reached_object = True
            if success:
                break
        
    success_rate /= args.num_evals
    if args.method_params['use_mask']:
        masks = np.stack(masks, axis=0)

    results['success_rate'] = success_rate
    results['masks'] = masks
    savefile = os.path.join(args.loadloc, f'results_domain_randomize_{args.domain_randomize}.npz')
    np.savez_compressed(savefile, success_rate=success_rate,
                        masks=masks)

    if args.save_video:
        video_writer.close()

    print('='*25)
    print(f'The policy achieved a success rate of {success_rate * 100} %')
    print('--')
    print(f'Saved {len(masks)} masks from the rollouts')
    print('--')
    print(f'Saved the results at this location: {savefile}')
    print('='*25)


if __name__ == '__main__':
    args = get_config()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ignore_args = {'loadloc', 'num_evals', 'render_mode', 'device', 'save_video', 'domain_randomize', 'inference_steps', 'n_rollout_actions'}
    
    trained_args_loc = os.path.join(args.loadloc, 'arguments.json')
    trained_args = json.load(open(trained_args_loc, 'r'))
    for k, v in trained_args.items():
        if k in ignore_args:
            continue
        if hasattr(args, k) and k != 'loadloc':
            setattr(args, k, v)
        else:
            print(f"Warning: Unrecognized arg '{k}' in {trained_args_loc}")
    if args.save_video:
        args.render = True
    
    if 'PickAndPlace' in args.env_name:
        args.time_horizon = 800
    else:
        args.time_horizon = 1000
    
    print('='*25)
    print(f'Evaluating the model in {args.loadloc}')
    print('='*25)

    evaluate(args)

    print('='*25)
    print('Done')
    print('='*25)