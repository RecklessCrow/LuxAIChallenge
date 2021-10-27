from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from constants import *
from lux_agent import LuxAgent
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback


def make_env(mode="train", model=None):
    return LuxEnvironment(
        configs=CONFIGS,
        learning_agent=LuxAgent(mode=mode, model=model),
        opponent_agent=LuxAgent(mode='inference', model=model)
    )


def train():
    """
    Main training loop
    :return:
    """

    # Create agents
    # trained_model = PPO.load(SAVED_MODEL_PATH)

    # Create Environment

    if NUM_ENVS > 1:
        dummy_env = make_vec_env(
            lambda: LuxEnvironment(
                configs=CONFIGS,
                learning_agent=LuxAgent(),
                opponent_agent=Agent(),
            ), NUM_ENVS)
    else:
        LuxEnvironment(
            configs=CONFIGS,
            learning_agent=LuxAgent(),
            opponent_agent=Agent(),
        )

    if NUM_ENVS > 1:
        train_env = make_vec_env(make_env, NUM_ENVS)
    else:
        train_env = make_env(model)

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
            replay_env=make_env(mode="inference", model=model),
            replay_num_episodes=5
        )
    )

    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    if NUM_ENVS > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        eval_env = make_vec_env(lambda: make_env(model, mode="inference"), NUM_EVAL_ENVS)

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
