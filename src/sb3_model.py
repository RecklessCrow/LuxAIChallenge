from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from torch import nn

from constants import *
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from src.lux_agent import LuxAgent


class LSTMFeatureExtractor(CombinedExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space)

        self.obs_q = None

    def forward(self, observations: dict) -> th.Tensor:
        observations = super().forward(observations)  # shape = (num_envs, obs_size)
        # print(observations.shape)

        return observations

        # if self.obs_q is None:
        #     self.obs_q = np.zeros([4] + list(observations.shape), dtype=np.float32)
        #
        # for idx in range(len(self.obs_q) - 1, 0, -1):
        #     self.obs_q[idx] = self.obs_q[idx - 1]
        #
        # self.obs_q[0] = observations
        #
        # return th.tensor(self.obs_q)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
            activation_fn=nn.GELU
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            # nn.LSTM(168, feature_dim),#nn.Tanh(),
            nn.Linear(feature_dim, last_layer_dim_pi), activation_fn(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), activation_fn()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.GELU,
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


def make_env(model=None):
    if os.path.exists(SAVED_MODEL_PATH):
        opponent_agent = LuxAgent(mode="inference", model=make_model(file_path=SAVED_MODEL_PATH))
    else:
        opponent_agent = Agent()

    return LuxEnvironment(
        configs=CONFIGS,
        learning_agent=LuxAgent(model=model),
        opponent_agent=opponent_agent
    )


def make_training_env(num_envs=NUM_ENVS):
    if num_envs > 1:
        train_env = make_vec_env(make_env, num_envs)
    else:
        train_env = make_env()

    return train_env


def make_model(env=None, file_path=None):
    if env is None and file_path is None:
        raise SyntaxError

    if file_path is not None:
        return PPO.load(file_path)

    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
    )

    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log=LOGS_PATH,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        batch_size=BATCH_SIZE,
        n_steps=NUM_STEPS,
        policy_kwargs=policy_kwargs
    )

    return model


def make_callbacks(model):
    # Create callbacks for logging
    callbacks = [
        SaveReplayAndModelCallback(
            save_freq=SAVE_FREQ,
            save_path=CHECKPOINT_PATH,
            name_prefix=TIME_STAMP,
            replay_env=LuxEnvironment(
                configs=CONFIGS,
                learning_agent=LuxAgent(mode="inference", model=model),
                opponent_agent=Agent()
            ),
            replay_num_episodes=5
        )
    ]

    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    if NUM_ENVS > 1:
        callbacks.append(
            EvalCallback(
                make_vec_env(lambda: make_env(model=model), NUM_EVAL_ENVS),
                best_model_save_path=CALLBACKS_PATH,
                log_path=CALLBACKS_PATH,
                eval_freq=NUM_STEPS * 2,  # Run it every 2 training iterations
                n_eval_episodes=NUM_EVAL_GAMES,
                deterministic=False,
                render=False
            )
        )

    return callbacks
