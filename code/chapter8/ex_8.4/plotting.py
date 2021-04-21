import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

with open('data/dyna_q_rewards.pickle', 'rb') as f:
    dyna_q_rewards = pickle.load(f)

with open('data/dyna_q_plus_rewards.pickle', 'rb') as f:
    dyna_q_plus_rewards = pickle.load(f)

with open('data/dyna_new_rewards.pickle', 'rb') as f:
    dyna_new_rewards = pickle.load(f)

cumulative_1 = np.cumsum(dyna_q_rewards[:10000])
cumulative_2 = np.cumsum(dyna_q_plus_rewards[:10000])
cumulative_3 = np.cumsum(dyna_new_rewards[:10000])

plt.figure(figsize=(10,5))
plt.plot(range(len(cumulative_1)), cumulative_1, label='dyna q', c='g', linewidth=0.75)
plt.plot(range(len(cumulative_2)), cumulative_2, label='dyna q +', c='r', linewidth=0.75)
plt.plot(range(len(cumulative_3)), cumulative_3, label='dyna new', c='b', linewidth=0.75)
plt.axvline(3000, label='wall switch', color='black', linestyle='--', linewidth=0.5)
plt.xlabel('timestep')
plt.ylabel('cumulative reward')
plt.legend()

plt.savefig('ex_8.4.png', dpi=300)
