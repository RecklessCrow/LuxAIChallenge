from multiprocessing import freeze_support
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from train import train


def main():
    train()
    # evaluate()


if __name__ == '__main__':
    freeze_support()
    main()

    # test_agent()
