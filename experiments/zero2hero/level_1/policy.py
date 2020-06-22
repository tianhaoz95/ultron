from abc import ABC, abstractmethod

class GamePolicy(ABC):
  @abstractmethod
  def compute_action(self, state, env):
    pass

class RandomPolicy(GamePolicy):
  def compute_action(self, state, env):
    return env.action_space.sample()