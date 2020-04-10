import gym

def toyrun():
  env = gym.make('CartPole-v0')
  for _ in range(20):
      observation = env.reset()
      for t in range(100):
          env.render()
          print(observation)
          action = env.action_space.sample()
          observation, _, done, _ = env.step(action)
          if done:
              print("Episode finished after {} timesteps".format(t+1))
  env.close()