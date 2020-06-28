from experiments.zero2hero.level_0.src.train import train_model
from experiments.zero2hero.level_0.src.test import test_model

def main():
  # train_model(visualize=True)
  train_model(static_plot=True)
  # test_model()

if __name__ == "__main__":
  main()

