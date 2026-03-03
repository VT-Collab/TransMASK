import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    task = 'Flip'
    main_dir = f'./results/{task}'
    total_runs = 3
    n = 4
    top_N = 3
    policy_type = 'diffusion'
    state_type = '_image'

    results = {'our': {'id': [],
                       'ood': []},
               'default': {'id': [],
                           'ood': []}}
    for i in tqdm(range(1, total_runs + 1)):
        our_results_loc = os.path.join(main_dir, f'{policy_type}{state_type}_mask_{i}')
        baseline_results_loc = os.path.join(main_dir, f'{policy_type}{state_type}_{i}')

        f1 = os.path.join(our_results_loc, 'results_domain_randomize_False.npz')
        results_our_id = np.load(f1)
        f2 = os.path.join(our_results_loc, 'results_domain_randomize_True.npz')
        results_our_ood = np.load(f2)

        f3 = os.path.join(baseline_results_loc, 'results_domain_randomize_False.npz')
        results_baseline_id = np.load(f3)
        f4 = os.path.join(baseline_results_loc, 'results_domain_randomize_True.npz')
        results_baseline_ood = np.load(f4)

        results['default']['id'].append(results_baseline_id['success_rate'])
        results['default']['ood'].append(results_baseline_ood['success_rate'])

        results['our']['id'].append(results_our_id['success_rate'])
        results['our']['ood'].append(results_our_ood['success_rate'])
        
    print('=' * 25)
    print('Baseline')
    print('-id  ', results['default']['id'])
    print('-ood ', results['default']['ood'])
    print('\n\n')
    print('Our')
    print('-id  ', results['our']['id'])
    print('-ood ', results['our']['ood'])
    print('=' * 25)