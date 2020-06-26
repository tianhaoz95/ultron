import config
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def collect_step(env, policy, buffer):
    next_state, reward, done, _ = env.step(action)
    buffer.record(reward, next_state)


class DqnAgent(object):
    def __init__(self):
        pass


class DqnBuffer(object):
    def __init__(self):
        pass


def train_model(
    num_iterations=config.default_num_iterations,
    collect_steps_per_iteration=config.default_collect_steps_per_iteration,
):
    env_name = config.default_env_name
    q_net = Sequential()
    train_env = gym.make(env_name)
    print('Action space: ')
    print(train_env.action_space)
    print('Observation space: ')
    print(train_env.observation_space)
    q_net.add(Dense(100, input_shape=(None, 6), activation='relu'))
    q_net.add(Dense(1))
    for eps_cnt in range(num_iterations):
        for step_cnt in range(collect_steps_per_iteration):
