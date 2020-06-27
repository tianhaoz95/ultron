import config
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def collect_step(env, policy, buffer):
    action = policy.action()
    next_state, reward, done, _ = env.step(action)
    buffer.record(reward, next_state)


class DqnAgent(object):
    def __init__(self):
        self.next_q_net = self.build_dqn_model()
        self.current_q_net = self.build_dqn_model()

    def build_dqn_model(self, observation_space, action_space):
        q_net = Sequential()
        q_net.add(Dense(100, input_shape=(None, observation_space), activation='relu'))
        q_net.add(Dense(action_space))
        return q_net


class DqnReplayBuffer(object):
    def __init__(self):
        pass


def train_model(
        num_iterations=config.default_num_iterations,
        collect_steps_per_iteration=config.default_collect_steps_per_iteration,
):
    env_name = config.default_env_name
    train_env = gym.make(env_name)
    print('Action space: ')
    print(train_env.action_space)
    print('Observation space: ')
    print(train_env.observation_space)
    for eps_cnt in range(num_iterations):
        for step_cnt in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, buffer)
