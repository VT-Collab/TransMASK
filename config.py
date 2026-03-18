import argparse
from ast import parse
from datetime import datetime


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='./datasets/demos_panda_pnp.hdf5')

    # Environment arguments
    parser.add_argument('--return_image_obs', action='store_true')
    parser.add_argument('--domain_randomize', action='store_true')
    parser.add_argument('--env_name', type=str, default=None)
    parser.add_argument('--render_mode', type=str, default='rgb_array')
    parser.add_argument('--num_distractors', type=int, default=1)
    
    # Training arguments
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--policy', type=str, default='deterministic')
    parser.add_argument('--use_ema_model', action='store_true')
    parser.add_argument('--ema_power', type=float, default=0.75)
    parser.add_argument('--ema_update_after_step', type=int, default=0)
    parser.add_argument('--ema_inv_gamma', type=float, default=1)
    parser.add_argument('--ema_min_value', type=float, default=0.)
    parser.add_argument('--ema_max_value', type=float, default=0.999)
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--activation', type=str, default='softmax')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[1024, 1024, 1024])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--obs_horizon', type=int, default=4)
    parser.add_argument('--pred_horizon', type=int, default=8)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--saveloc', type=str, default='./results/')
    default_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    parser.add_argument('--savename', type=str, default=default_name)

    # Method arguments
    parser.add_argument('--method', type=str, default='default', choices=['default', 'vae', 'our'])

    # Evaluating arguments
    parser.add_argument('--loadloc', type=str, default=None)
    parser.add_argument('--num_evals', type=int, default=10)
    parser.add_argument('--time_horizon', type=int, default=300)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--inference_steps', type=int, default=25)
    parser.add_argument('--n_rollout_actions', type=int, default=1)
    
    args = parser.parse_args()

    # Update arguments based on method
    use_mask = False
    use_vae = False
    if args.method == 'default':
        pass
    elif args.method == 'vae':
        use_vae = True
    elif args.method == 'our':
        use_mask = True
    args.method_params = {'use_mask': use_mask,
                          'use_vae': use_vae}
    if args.policy == 'diffusion':
        args.sequential = True
        args.use_ema_model = True
    return args
