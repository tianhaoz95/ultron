import random
import gym
import tensorflow as tf
import numpy as np
from . import config
from tensorflow.keras.models import Sequential
from collections import deque
from tensorflow.keras.layers import Dense


def collect_steps(env, policy, buffer, max_steps):
    state = env.reset()
    for step_cnt in range(max_steps):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        buffer.record(state, reward, next_state, action, done)
        if done:
            state = env.reset()
        else:
            state = next_state


def compute_avg_reward(env, policy, num_episodes):
    total_return = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return


class DqnAgent(object):
    def __init__(self, state_space, action_space, gamma, lr, verbose):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma
        self.verbose = verbose
        if self.verbose:
            print('Construct DQN agent with: ')
            print('Action space: ')
            print(action_space)
            print('State space: ')
            print(state_space)
        self.q_net = self._build_dqn_model(state_space=state_space, action_space=action_space, lr=lr)

    @staticmethod
    def _build_dqn_model(state_space, action_space, lr):
        q_net = Sequential()
        q_net.add(Dense(128, input_dim=state_space, activation='relu'))
        q_net.add(Dense(64, activation='relu'))
        q_net.add(Dense(action_space, activation='relu'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss='mse')
        return q_net

    def train(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, batch_size):
        current_q = self.q_net(state_batch).numpy()
        next_q = self.q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        if self.verbose:
            print('reward batch shape: ', reward_batch.shape)
            print('next Q shape: ', next_q.shape)
            print('next state batch shape: ', next_state_batch.shape)
            print('max next Q shape: ', max_next_q.shape)
        for batch_idx in range(batch_size):
            if done_batch[batch_idx]:
                current_q[batch_idx][action_batch[batch_idx]] = reward_batch[batch_idx]
            else:
                current_q[batch_idx][action_batch[batch_idx]] = reward_batch[batch_idx] + self.gamma * max_next_q[
                    batch_idx]
        loss = self.q_net.train_on_batch(x=state_batch, y=current_q)
        return loss

    def random_policy(self, state):
        return np.random.randint(0, self.action_space)

    def policy(self, state):
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        optimal_action = np.argmax(action_q.numpy()[0], axis=0)
        if self.verbose:
            print('state: ', state)
            print('state_input: ', state_input)
            print('action Q: ', action_q)
            print('optimal action: ', optimal_action)
        return optimal_action


class DqnReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)

    def record(self, state, reward, next_state, action, done):
        self.experiences.append((state, next_state, action, reward, done))

    def sample_batch(self, batch_size):
        sampled_batch = random.sample(self.experiences, batch_size)
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for record in sampled_batch:
            state_batch.append(record[0])
            next_state_batch.append(record[1])
            action_batch.append(record[2])
            reward_batch.append(record[3])
            done_batch.append(record[4])
        return np.array(state_batch), np.array(next_state_batch), np.array(action_batch), np.array(
            reward_batch), np.array(done_batch)


def train_model(
        num_iterations=config.default_num_iterations,
        collect_steps_per_iteration=config.default_collect_steps_per_iteration,
        batch_size=config.default_batch_size,
        max_replay_history=config.default_max_replay_history,
        gamma=config.default_gamma,
        eval_eps=config.default_eval_eps,
        learning_rate=config.default_learning_rate,
        verbose=False,
):
    env_name = config.default_env_name
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)
    agent = DqnAgent(state_space=train_env.observation_space.shape[0], action_space=train_env.action_space.n,
                     gamma=gamma, verbose=verbose, lr=learning_rate)
    benchmark_reward = compute_avg_reward(eval_env, agent.random_policy, eval_eps)
    for eps_cnt in range(num_iterations):
        buffer = DqnReplayBuffer(max_size=max_replay_history)
        collect_steps(train_env, agent.policy, buffer, batch_size)
        collect_steps(train_env, agent.policy, buffer, collect_steps_per_iteration)
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = buffer.sample_batch(
            batch_size=batch_size)
        loss = agent.train(state_batch=state_batch, next_state_batch=next_state_batch, action_batch=action_batch,
                           reward_batch=reward_batch, done_batch=done_batch, batch_size=batch_size)
        avg_reward = compute_avg_reward(eval_env, agent.policy, eval_eps)
        print('Episode {0}/{1}({2}%) finished with loss {3} and average reward {4} against benchmark reward {5}'.format(
            eps_cnt, num_iterations,
            round(eps_cnt / num_iterations * 100.0, 2),
            loss,
            avg_reward, benchmark_reward))
    train_env.close()
    eval_env.close()
