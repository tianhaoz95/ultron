import config
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.utils.common import Checkpointer
from tensorflow.keras.optimizers import Adam


def test_drive(env_name):
    test_env = suite_gym.load(env_name)
    print('Action spec:')
    print(test_env.action_spec())
    test_env.reset()
    time_step = test_env.reset()
    print("Time step: ")
    print(time_step)
    action = np.array(1, dtype=np.int32)
    next_time_step = test_env.step(action)
    print('Next time step:')
    print(next_time_step)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


def train_model(
    num_iterations=config.default_num_iterations,
    collect_steps_per_iteration=config.default_collect_steps_per_iteration,
    replay_buffer_max_length=config.default_replay_buffer_max_length,
    batch_size=config.default_batch_size,
    learning_rate=config.default_learning_rate,
    log_interval=config.default_log_interval,
    num_eval_episodes=config.default_num_eval_episodes,
    eval_interval=config.default_eval_interval,
    checkpoint_saver_directory=config.default_checkpoint_saver_directory,
    model_saver_directory=config.default_model_saver_directory,
    visualize=False,
    static_plot=False,
):
    if visualize:

        import streamlit as st
    env_name = 'CartPole-v0'
    # test_drive(env_name=env_name)
    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    fc_layer_params = (100,)
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    optimizer = Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    agent.initialize()
    eval_policy = agent.policy
    model_saver = PolicySaver(policy=agent.policy)
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec())
    sample_avg_return = compute_avg_return(
        eval_env, random_policy, num_eval_episodes)
    print("Sample average return with random policy: ")
    print(sample_avg_return)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)
    train_checkpointer = Checkpointer(
        ckpt_dir=checkpoint_saver_directory,
        max_to_keep=12,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
    )
    train_checkpointer.initialize_or_restore()
    # collect_data(train_env, random_policy, replay_buffer, steps=100)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    print("Dataset sample: ")
    print(dataset)
    # training loop
    iterator = iter(dataset)
    agent.train_step_counter.assign(0)
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = []
    loss = []
    max_avg_return = 0.0
    if visualize:
        st.title(body="Average return") # pylint: disable=no-value-for-parameter
        return_chart = st.line_chart(returns)
        st.title(body="Loss") # pylint: disable=no-value-for-parameter
        loss_chart = st.line_chart(loss)
    for _ in range(num_iterations):
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        if visualize:
            loss_chart.add_rows([train_loss])
        step = agent.train_step_counter.numpy()
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        if visualize:
            return_chart.add_rows([avg_return])
        print('step={0}/{1}: avg_return={2}, loss={3}'.format(
            step, num_iterations, avg_return, train_loss))
        if avg_return > max_avg_return:
            avg_return = max_avg_return
            model_saver.save(model_saver_directory)
        train_checkpointer.save(step)
        returns.append(avg_return)
        if static_plot:
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.plot(avg_return)
            ax2.plot(returns)
    print('Done')
