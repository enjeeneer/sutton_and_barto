from models import Env, Agent
import numpy as np
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    env_1 = Env(b=1)
    env_3 = Env(b=3)

    agent_on_policy_1 = Agent(on_policy=True)
    agent_uniform_1 = Agent(uniform=True)
    agent_on_policy_3 = Agent(on_policy=True)
    agent_uniform_3 = Agent(uniform=True)

    e_updates = 200000
    e_interval = 1000

    b1_on_policy_v = []
    b1_uniform_v = []
    b3_on_policy_v = []
    b3_uniform_v = []

    e_update_intervals = np.arange(0, e_updates, e_interval)

    with open('data/e_update_intervals.pickle', 'wb') as f:
        pickle.dump(e_update_intervals, f)

    print("Beginning experiment with b=3 and uniform agent")
    # run b = 3 experiment uniform
    while agent_uniform_3.expected_updates < e_updates:
        for state in tqdm(env_3.states):
            action = np.random.choice(agent_uniform_3.actions)
            state_, reward = env_3.step(state, uniform=True)
            agent_uniform_3.learn(state_, state, action, reward, env_3.b, uniform=True)
            if agent_uniform_3.expected_updates % e_interval == 0:
                v = agent_uniform_3.evaluate(env=env_3)
                b3_uniform_v.append(v)

    with open('data/b3_uniform_v.pickle', 'wb') as f:
        pickle.dump(b3_uniform_v, f)

    print("Beginning experiment with b=1 and uniform agent")
    # run b = 1 experiment uniform
    while agent_uniform_1.expected_updates < e_updates:
        for state in tqdm(env_1.states):
            action = np.random.choice(agent_uniform_1.actions)
            state_, reward = env_1.step(state, uniform=True)
            agent_uniform_1.learn(state_, state, action, reward, env_1.b, uniform=True)
            if agent_uniform_1.expected_updates % e_interval == 0:
                v = agent_uniform_1.evaluate(env=env_1)
                b1_uniform_v.append(v)

    with open('data/b1_uniform_v.pickle', 'wb') as f:
        pickle.dump(b1_uniform_v, f)

    print("Beginning experiment with b=1 and on-policy agent")
    # run b = 1 experiment on-policy
    for i in tqdm(range(e_updates)):
        state = env_1.reset()
        done = False
        while not done:
            a_idx = agent_on_policy_1.choose_action(state)
            state_, reward, done = env_1.step(state, on_policy=True)
            agent_on_policy_1.learn(state_, state, a_idx, reward, env_1.b, on_policy=True)
            if agent_on_policy_1.expected_updates % e_interval == 0:
                v = agent_on_policy_1.evaluate(env=env_1)
                b1_on_policy_v.append(v)

    with open('data/b1_on_policy_v.pickle', 'wb') as f:
        pickle.dump(b1_on_policy_v, f)

    print("Beginning experiment with b=3 and on-policy agent")
    # run b = 3 experiment on-policy
    for i in tqdm(range(e_updates)):
        state = env_3.reset()
        done = False
        while not done:
            a_idx = agent_on_policy_3.choose_action(state)
            state_, reward, done = env_3.step(state, on_policy=True)
            agent_on_policy_3.learn(state_, state, a_idx, reward, env_3.b, on_policy=True)
            if agent_on_policy_3.expected_updates % e_interval == 0:
                v = agent_on_policy_3.evaluate(env=env_3)
                b3_on_policy_v.append(v)

    with open('data/b3_on_policy_v.pickle', 'wb') as f:
        pickle.dump(b3_on_policy_v, f)
