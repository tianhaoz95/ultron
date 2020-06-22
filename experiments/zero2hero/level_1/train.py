import gym
from memory import GameMemory
from policy import RandomPolicy
from run_trial import run_trial

def train(eps_cnt):
  env = gym.make('CartPole-v0')
  for _ in range(eps_cnt):
    memory = GameMemory()
    policy = RandomPolicy()
    run_trial(policy=policy, env=env, memory=memory)