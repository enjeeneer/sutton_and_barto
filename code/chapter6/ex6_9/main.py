import numpy as np
import gym
import gym_gridworlds
from tqdm import tqdm
import itertools

class agent():
    def __init__(self, env_size, action_space, epsilon=0.1, alpha=0.5):
        self.env_size = env_size
        self.action_space = action_space
        self.q = np.zeros((env_size[0], env_size[1], action_space), dtype='float')

        self.epsilon = epsilon
        self.alpha = alpha

    def choose_action(self, state, greedy=False):
        if greedy:
            a_s = self.q[state[0], state[1],]
            action = np.argmax(a_s)
        elif np.random.random_sample() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            a_s = self.q[state[0], state[1],]
            action = np.argmax(a_s)
        return action

    def learn(self, observation, reward, state, action):
        action_ = self.choose_action(observation)
        q_ = self.q[observation[0], observation[1], action_]
        q = self.q[state[0], state[1], action]
        self.q[state[0], state[1], action] = q + self.alpha*(reward + q_ - q)


# amend for new moves
kings_moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 1), 5: (1, 1), 6: (1, -1), 7: (-1, -1)}
kings_moves_plus = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 1), 5: (1, 1), 6: (1, -1), 7: (-1, -1), 8: (0, 0)}

# create gridworld and agent
env = gym.make('WindyGridworld-v0')
agent = agent(env_size=(7,10), action_space=len(kings_moves))

env.moves = kings_moves
env.action_space.n = len(env.moves)

state = env.reset()
for e in tqdm(range(10000)):
    t = 0
    done = False
    while not done:
        action = agent.choose_action(state)
        observation, reward, done, info = env.step(action)
        agent.learn(observation, reward, state, action)
        state = observation
        t += 1
        if done:
            print(f"Episode {e} finished after {t} timesteps")
            state = env.reset()
            break

# plot optimal policy
v_star = np.zeros((7, 10), dtype='int')
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state, greedy=True)
    observation, reward, done, info = env.step(action)
    v_star[observation[0], observation[1]] = 1
    state = observation
    if done:
        print(f"Episode {e} finished after {t} timesteps")
        state = env.reset()
        break

print(v_star)







# {'height': 7, 'width': 10, 'action_space': Discrete(4), 'observation_space': Tuple(Discrete(7), Discrete(10)), 'moves': {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}, 'S': (3, 0), 'spec': EnvSpec(WindyGridworld-v0)}
