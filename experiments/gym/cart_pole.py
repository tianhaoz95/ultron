import gym
env = gym.make('CartPole-v0')
print('action space: ', env.action_space)
print('observation space', env.observation_space)
for i_episode in range(10):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print('observation: ', observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            # print('Episode finished after {} timesteps'.format(t+1))
            break
env.close()