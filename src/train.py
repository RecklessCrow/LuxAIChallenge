from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv

from constants import *
# from sb3_model import make_model, make_training_env, make_callbacks
from stable_baselines import PPO2

from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import SaveReplayAndModelCallback, LuxEnvironment
from src.lux_agent import LuxAgent
from stable_baselines.common.vec_env import DummyVecEnv

def make_env(model=None):
    if os.path.exists(SAVED_MODEL_PATH):
        opponent_agent = LuxAgent(mode="inference", model=model)
    else:
        opponent_agent = Agent()

    return LuxEnvironment(
        configs=CONFIGS,
        learning_agent=LuxAgent(model=model),
        opponent_agent=LuxAgent(mode="inference", model=model))


def make_training_env(num_envs=NUM_ENVS):
    if num_envs > 1:
        train_env = make_vec_env(make_env, num_envs)
    else:
        train_env = DummyVecEnv([lambda: make_env])

    return train_env


def train():
    """
    Main training loop
    :return:
    """

    train_env = SubprocVecEnv([make_env for i in range(NUM_ENVS)])
    # model = make_model(env=train_env)

    train_env = VecFrameStack(train_env, n_stack=4)
    model = PPO2("MlpPolicy", train_env,
                 verbose=1,
                 tensorboard_log=LOGS_PATH,
                 learning_rate=LEARNING_RATE,
                 gamma=GAMMA,
                 n_steps=NUM_STEPS,
                 )

    callbacks = []

    # Replay & Checkpoint
    player_replay = LuxAgent(mode="inference", model=model)
    opponent_replay = Agent()
    callbacks.append(
        SaveReplayAndModelCallback(
            save_freq=SAVE_FREQ,
            save_path=CHECKPOINT_PATH,
            name_prefix=TIME_STAMP,
            replay_env=LuxEnvironment(
                configs=CONFIGS,
                learning_agent=player_replay,
                opponent_agent=opponent_replay
            ),
            replay_num_episodes=5
        )
    )

    if NUM_ENVS > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        eval_env = make_vec_env(make_env, 1)
        eval_env = VecFrameStack(eval_env, n_stack=4)

        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=CALLBACKS_PATH,
                log_path=CALLBACKS_PATH,
                eval_freq=NUM_STEPS * 10,  # Run it every 2 training iterations
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
