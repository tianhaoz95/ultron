import gym
import argparse
from master import MasterAgent
from toy import toyrun

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'Cartpole.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--toy', dest='toy', action='store_true',
                    help='Just play the cartpole environment')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=2000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()

if __name__ == '__main__':
  print(args)
  if args.toy:
    toyrun()
  else:
    master = MasterAgent(args)
    if args.train:
      master.train()
    else:
      master.play()