from multiprocessing import freeze_support

import torch

from train import train


def main():
    train()
    # evaluate()


if __name__ == '__main__':
    device = torch.device("cpu")
    freeze_support()
    main()
