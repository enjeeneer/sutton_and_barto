import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data/final_v_functions.pickle', 'rb') as f:
    v_functions = pickle.load(f)

with open('data/sweeps.pickle', 'rb') as f:
    sweeps = pickle.load(f)

with open('data/policy.pickle', 'rb') as f:
    policy = pickle.load(f)

ph = ['0.25', '0.55']

for p in ph:

    sweep_p = []
    policy_p = []
    for arr in sweeps[p]:
        sweep_p.append(arr.flatten())
    for arr in policy[p]:
        policy_p.append(arr)

    x = np.arange(0,101,1)
    x_pol = np.arange(1,100,1)

    fig = plt.figure(figsize=(15,10))
    fig.suptitle(f'$P_h({p})$', fontsize=16)

    ax1 = fig.add_subplot(211)
    ax1.title.set_text('Value Function Approximation per Sweep')

    i = 1
    for arr in sweep_p:
        ax1.plot(x, arr, label='sweep: {}'.format(i))
        i += 1

    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.title.set_text('Optimal Policy')
    ax2.plot(x_pol, policy_p)

    plt.savefig('images/p_{}.png'.format(p), dpi=300)
