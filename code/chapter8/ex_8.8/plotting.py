import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('data/e_update_intervals.pickle', 'rb') as f:
    e_update = pickle.load(f)

with open('data/b1_uniform_v.pickle', 'rb') as f:
    b1_uniform_v = pickle.load(f)

with open('data/b3_uniform_v.pickle', 'rb') as f:
    b3_uniform_v = pickle.load(f)

with open('data/b1_on_policy_v.pickle', 'rb') as f:
    b1_on_policy_v = pickle.load(f)

with open('data/b3_on_policy_v.pickle', 'rb') as f:
    b3_on_policy_v = pickle.load(f)

plt.figure(figsize=(10,5))
# plt.plot(e_update, b1_on_policy_v, label='on-policy', c='g', linewidth=0.75)
plt.plot(e_update, b1_uniform_v, label='on-policy', c='r', linewidth=0.75)
plt.xlabel('Computation time, in expected updates')
plt.ylabel('Value of start under greedy policy')
plt.legend()
plt.savefig('ex8.8_1.png', dpi=300)

plt.figure(figsize=(10,5))
plt.plot(e_update, b3_on_policy_v, label='on-policy', c='g', linewidth=0.75)
plt.plot(e_update, b3_uniform_v, label='on-policy', c='r', linewidth=0.75)
plt.xlabel('Computation time, in expected updates')
plt.ylabel('Value of start under greedy policy')
plt.legend()
plt.savefig('ex8.8_2.png', dpi=300)
