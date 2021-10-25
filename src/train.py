import copy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from constants import *
from lux_agent import LuxAgent
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default


def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.
    :param local_env: (LuxEnvironment) the environment
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


def train():
    """
    Main training loop
    :return:
    """

    # Create agents
    player = LuxAgent(mode="train")
    opponent = Agent()

    # Create Environment
    configs = LuxMatchConfigs_Default

    if NUM_ENVS > 1:
        train_env = SubprocVecEnv([make_env(
            LuxEnvironment(
                configs=configs,
                learning_agent=copy.deepcopy(player),
                opponent_agent=copy.deepcopy(opponent)
            ), i) for i in range(NUM_ENVS)]
        )
    else:
        train_env = LuxEnvironment(
            configs=configs,
            learning_agent=player,
            opponent_agent=opponent
        )

    # Create Model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOGS_PATH,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        batch_size=BATCH_SIZE,
        n_steps=NUM_STEPS
    )

    # Create callbacks for logging
    callbacks = []

    # Replay & Checkpoint
    player_replay = LuxAgent(mode="inference", model=model)
    opponent_replay = Agent()
    callbacks.append(
        SaveReplayAndModelCallback(
            save_freq=100000,
            save_path=CHECKPOINT_PATH,
            name_prefix=TIME_STAMP,
            replay_env=LuxEnvironment(
                configs=configs,
                learning_agent=player_replay,
                opponent_agent=opponent_replay
            ),
            replay_num_episodes=5
        )
    )

    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    if NUM_ENVS > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        env_eval = SubprocVecEnv([make_env(LuxEnvironment(
            configs=configs,
            learning_agent=LuxAgent(mode="train"),
            opponent_agent=opponent), i) for i in range(4)]
        )

        callbacks.append(
            EvalCallback(
                env_eval,
                best_model_save_path=f'./logs_{TIME_STAMP}/',
                log_path=f'./logs_{TIME_STAMP}/',
                eval_freq=NUM_STEPS * 2,  # Run it every 2 training iterations
                n_eval_episodes=30,  # Run 30 games
                deterministic=False, render=False
            )
        )

    print("Training Model...")
    model.learn(
        total_timesteps=MAX_STEPS,
        callback=callbacks
    )

    print("Saving Model...")
    if not os.path.exists(MODEL_PATH):
        model.save(path=MODEL_PATH)

    print("Done training model.")

    # Inference the model
    # eval_env = LuxEnvironment()

    obs = train_env.reset()
    for i in range(600):
        action_code, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = train_env.step(action_code)
        if i % 5 == 0:
            print("Turn %i" % i)
            train_env.render()

        if done:
            print("Episode done, resetting.")
            obs = train_env.reset()
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