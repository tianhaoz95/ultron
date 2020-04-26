import argparse
import toy
import master

parser = argparse.ArgumentParser(description='Play the break out game')
parser.add_argument('--toy', dest='toy', action='store_true',
                    help='Just play the break out environment.')
parser.add_argument('--play', dest='play', action='store_true',
                    help='Play the game with specified model.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--max-eps', default=2000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between a3c and random.')
args = parser.parse_args()


def main():
    print('Hello World')
    if (args.toy):
        toy.play()
    else:
      planner = master.Master(args)
      if args.train:
        planner.train()
    print('All done!')


if __name__ == '__main__':
    main()
