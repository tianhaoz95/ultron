import gym

class BaselineModel:
  """
  Baseline Model is an agent that plays the
  game randomly.
  """
  def __init__(self, env_name, max_eps):
    self.env = gym.make(env_name)
    self.max_eps = max_eps

  def run(self):
    total_reward = 0
    for eps in range(self.max_eps):
      done = False
      step = 0
      eps_reward = 0
      self.env.reset()
      while not done:
        action = self.env.action_space.sample()
        _, reward, done, _ = self.env.step(action)
        step += 1
        eps_reward += reward
      total_reward += eps_reward
    avg_reward = total_reward / float(self.max_eps)
    print('Average reward: ', avg_reward)
    return avg_reward
    