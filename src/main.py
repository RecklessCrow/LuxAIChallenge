import torch

from train import train, evaluate
from multiprocessing import freeze_support


def main():
    train()
    # evaluate()


if __name__ == '__main__':
    device = torch.device("cpu")
    freeze_support()
    main()
