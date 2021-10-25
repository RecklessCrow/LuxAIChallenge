from train import train, evaluate
from multiprocessing import freeze_support


def main():
    train()
    # evaluate()


if __name__ == '__main__':
    freeze_support()
    main()
