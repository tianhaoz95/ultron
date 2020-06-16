import gym
import baseline

class Master():
  def __init__(self, args):
    self.args = args
    self.game_name = 'Breakout-ram-v0'
    env = gym.make(self.game_name)
    self.state_size = env.observation_space.shape[0]
    self.action_size = env.action_space.n
  
  def train(self):
    if self.args.algorithm == 'random':
      random_agent = baseline.BaselineModel(self.game_name, self.args.max_eps)
      random_agent.run()
      return