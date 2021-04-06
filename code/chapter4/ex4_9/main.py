import numpy as np
from models import environment, agent
import pickle

if __name__ == '__main__':
    env = environment()
    agent = agent()

    phs = [0.25, 0.55]

    final_vf = {}
    sweeps = {}
    policy = {}

    for ph in phs:
        v_func, sweep = agent.value_iteration(ph, env)
        stakes = agent.find_policy(v_func, env, ph)
        final_vf[str(ph)] = v_func
        sweeps[str(ph)] = sweep
        policy[str(ph)] = stakes

    with open('data/final_v_functions.pickle', 'wb') as f:
        pickle.dump(final_vf, f)

    with open('data/sweeps.pickle', 'wb') as f:
        pickle.dump(sweeps, f)

    with open('data/policy.pickle', 'wb') as f:
        pickle.dump(policy, f)
