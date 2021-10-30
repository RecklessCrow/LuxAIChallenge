from multiprocessing import freeze_support

from train import train


def main():
    train()
    # evaluate()


def test_agent():
    from lux_agent import LuxAgent
    from train import make_env

    env = make_env()
    env.reset()

    temp_agent = LuxAgent()
    (unit, city_tile, team, is_new_turn) = next(env.match_generator)
    obs = temp_agent.get_observation(env.game, unit, city_tile, team, is_new_turn)
    print(obs)


if __name__ == '__main__':
    freeze_support()
    main()

    # test_agent()
