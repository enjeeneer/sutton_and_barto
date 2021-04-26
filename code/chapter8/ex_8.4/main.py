from models import Environment, Agent
from tqdm import tqdm
import pickle

if __name__ == '__main__':
    env_dyna_q = Environment()
    env_dyna_q_plus = Environment()
    env_dyna_new = Environment()


    dyna_q = Agent(env_dyna_q.action_space, dyna_q=True)
    dyna_q_plus = Agent(env_dyna_q_plus.action_space, dyna_q_plus=True)
    dyna_new = Agent(env_dyna_new.action_space, dyna_new=True)

    dyna_q_reward = []
    dyna_q_plus_reward = []
    dyna_new_reward = []

    # run dyna_q
    for i in tqdm(range(1000)):
        done = False
        state = env_dyna_q.reset()
        while not done:
            a_idx = dyna_q.choose_action(state)
            state_, reward, done = env_dyna_q.step(state, a_idx)
            dyna_q.learn(state_, state, a_idx, reward)
            if i > 0:
                dyna_q.plan()
            state = state_

            dyna_q_reward.append(reward)

    # store binary data
    with open('data/dyna_q_rewards.pickle', 'wb') as f:
        pickle.dump(dyna_q_reward, f)

    #run dyna_q_plus
    for i in tqdm(range(1000)):
        done = False
        state = env_dyna_q_plus.reset()
        while not done:
            a_idx = dyna_q_plus.choose_action(state)
            state_, reward, done = env_dyna_q_plus.step(state, a_idx)
            dyna_q_plus.learn(state_, state, a_idx, reward)
            if i > 0:
                dyna_q_plus.plan()
            state = state_

            dyna_q_plus_reward.append(reward)

    # store binary data
    with open('data/dyna_q_plus_rewards.pickle', 'wb') as f:
        pickle.dump(dyna_q_plus_reward, f)

    # run dyna_new
    for i in tqdm(range(1000)):
        done = False
        state = env_dyna_new.reset()
        while not done:
            a_idx = dyna_new.choose_action(state)
            state_, reward, done = env_dyna_new.step(state, a_idx)
            dyna_new.learn(state_, state, a_idx, reward)
            if i > 0:
                dyna_new.plan()
            state = state_

            dyna_new_reward.append(reward)

    # store binary data
    with open('data/dyna_new_rewards.pickle', 'wb') as f:
        pickle.dump(dyna_new_reward, f)
