class GameMemory:
  def __init__(self):
    self.rewards = []
    self.states = []
    self.actions = []

  def record(self, reward, state, action):
    self.rewards.append(reward)
    self.states.append(state)
    self.actions.append(action)
