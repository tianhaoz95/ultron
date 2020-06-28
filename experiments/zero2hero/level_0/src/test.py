import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

def test_model():
  env_name = 'CartPole-v0'
  env = suite_gym.load(env_name)
  tf_env = tf_py_environment.TFPyEnvironment(env)
  time_step = tf_env.reset()
  saved_policy  = tf.saved_model.load("./models")
  policy_state = saved_policy.get_initial_state(batch_size=3)
  total_reward = 0.0
  while not time_step.is_last():
    env.render()
    policy_step = saved_policy.action(time_step, policy_state)
    policy_state = policy_step.state
    time_step = tf_env.step(policy_step.action)
    total_reward += time_step.reward
  env.close()
  print("Done with reward {0}".format(total_reward))