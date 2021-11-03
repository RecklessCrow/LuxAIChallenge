from multiprocessing import freeze_support

from train import train


def main():
    train()
    # evaluate()


if __name__ == '__main__':
    freeze_support()
    main()

    # test_agent()
