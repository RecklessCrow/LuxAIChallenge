import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from constants import *
from lux_agent import LuxAgent
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback


def action_mask_fn(env: LuxEnvironment):
    valid_actions = np.zeros(env.action_space.n)

    city_tile = env.last_observation_object[1]
    if city_tile is not None:
        city_count = 0

        for city in env.game.cities.values():
            if city.team == city_tile.team:
                city_count += 1

        if city_count > len(env.game.get_teams_units(city_tile.team)):
            valid_actions[:2] = True  # can only build a unit if num citits > num units

        valid_actions[2] = True  # city may always research

    else:
        unit = env.last_observation_object[0]

        valid_actions[:5] = True  # movement. Check for if unit is on map boarder?

        # ToDo
        #  valid_actions[5] if worker adjacent
        #  valid_actions[6] if cart adjacent
        valid_actions[5:7] = True

        if unit.is_worker():
            if unit.can_build(env.game.map):
                valid_actions[7] = True

            cell = env.game.map.get_cell_by_pos(unit.pos)
            if cell.road > CONFIGS["parameters"]["MIN_ROAD"]:
                valid_actions[8] = True

    return valid_actions


def make_env(mode="train"):
    return ActionMasker(LuxEnvironment(
        configs=CONFIGS,
        learning_agent=LuxAgent(mode=mode),
        opponent_agent=Agent()
    ), action_mask_fn)


def train():
    """
    Main training loop
    :return:
    """

    # Create agents
    # trained_model = PPO.load(SAVED_MODEL_PATH)

    # Create Environment

    if NUM_ENVS > 1:
        train_env = make_vec_env(make_env, NUM_ENVS)
    else:
        train_env = make_env()

    # Create Model
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOGS_PATH,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        batch_size=BATCH_SIZE,
        n_steps=NUM_STEPS,
    )

    # Create callbacks for logging
    callbacks = []

    # Replay & Checkpoint
    callbacks.append(
        SaveReplayAndModelCallback(
            save_freq=SAVE_FREQ,
            save_path=CHECKPOINT_PATH,
            name_prefix=TIME_STAMP,
            replay_env=make_env(mode="inference"),
            replay_num_episodes=5
        )
    )

    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    if NUM_ENVS > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        eval_env = make_vec_env(make_env, NUM_EVAL_ENVS)

        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=CALLBACKS_PATH,
                log_path=CALLBACKS_PATH,
                eval_freq=NUM_STEPS * 2,  # Run it every 2 training iterations
                n_eval_episodes=NUM_EVAL_GAMES,
                deterministic=False,
                render=False
            )
        )

    print("Training Model...")
    model.learn(
        total_timesteps=TRAINING_STEPS,
        callback=callbacks
    )

    print("Saving Model...")
    if not os.path.exists(MODEL_PATH):
        model.save(path=MODEL_PATH)

    print("Done training model.")

    # evaluate(model)


def evaluate(model=None):
    player = LuxAgent(model=model)
    opponent = LuxAgent(model=PPO.load(SAVED_MODEL_PATH))

    configs = LuxMatchConfigs_Default

    eval_env = LuxEnvironment(
        configs=configs,
        learning_agent=player,
        opponent_agent=opponent
    )

    obs = eval_env.reset()
    for i in range(600):
        action_code, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = eval_env.step(action_code)
        if i % 5 == 0:
            print(f"Turn {i}")
            eval_env.render()

        if done:
            print("Episode done, resetting.")
            obs = eval_env.reset()
    print("Done")

    '''
    # Learn with self-play against the learned model as an opponent now
    print("Training model with self-play against last version of model...")
    player = AgentPolicy(mode="train")
    opponent = AgentPolicy(mode="inference", model=model)
    env = LuxEnvironment(configs, player, opponent)
    model = PPO("MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./lux_tensorboard/",
        learning_rate = 0.0003,
        gamma=0.999,
        gae_lambda = 0.95
    )
    model.learn(total_timesteps=2000)
    env.close()
    print("Done")
    '''
