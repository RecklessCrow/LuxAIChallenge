import os
from datetime import datetime

import numpy as np

from luxai2021.game.constants import Constants, LuxMatchConfigs_Default
from luxai2021.game.game_constants import GAME_CONSTANTS

CONFIGS = LuxMatchConfigs_Default

RESOURCE_LIST = [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL, Constants.RESOURCE_TYPES.URANIUM]

MAX_DAYS = GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

GAME_STATE_CATEGORIES = [
    'is_night', '%_of_cycle_passed', '%_game_complete',
    'research_points', 'coal_is_researched', 'uranium_is_researched',
    'worker_cap_reached', "workers_self", "workers_opponent", "cities_self", "cities_opponent"
]
GAME_STATE_LEN = len(GAME_STATE_CATEGORIES)
GAME_STATE_IDX_DICT = dict(zip(GAME_STATE_CATEGORIES, np.arange(0, len(GAME_STATE_CATEGORIES))))

NUM_UNIT_OBSERVATIONS = 3
UNIT_CATEGORIES = ['team', 'city', 'worker', 'cart', 'inventory', 'x', 'y', 'angle_to_controlled', 'dist_to_controlled']
UNIT_LEN = len(UNIT_CATEGORIES)
UNIT_IDX_DICT = dict(zip(UNIT_CATEGORIES, np.arange(0, len(UNIT_CATEGORIES))))

NUM_RESOURCE_OBSERVATIONS = 5
RESOURCE_CATEGORIES = ['wood', 'coal', 'uranium', 'amount', 'x', 'y', 'angle_to_controlled', 'dist_to_controlled']
RESOURCE_LEN = len(RESOURCE_CATEGORIES)
RESOURCE_IDX_DICT = dict(zip(RESOURCE_CATEGORIES, np.arange(0, RESOURCE_LEN)))

# Observation Constants
RESEARCH_FOR_COAL = 50
MAX_RESEARCH = 200
STARTING_CITIES = 1
STARTING_UNITS = 1
TEAMS = [0, 1]
POSITION_SCALE = 32 // 2
MAX_WORKERS = 30
MAX_CITIES = MAX_WORKERS

NUM_STEPS_IN_DAY = 30
NUM_STEPS_IN_NIGHT = 10

# Reward Constants
GAME_WIN = 5

CITY_MADE = 2

COAL_UNLOCKED = 5
URANIUM_UNLOCKED = 5

FUEL_DEPOSITED_REWARD_MODIFIER = 0.001
WOOD_GATHERED_REWARD_MODIFIER = 0.001
COAL_GATHERED_REWARD_MODIFIER = WOOD_GATHERED_REWARD_MODIFIER * 10
URANIUM_GATHERED_REWARD_MODIFIER = WOOD_GATHERED_REWARD_MODIFIER * 40

# Hyper Parameters
LEARNING_RATE = 1e-4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # How much to value rewards in the future
BATCH_SIZE = 2 ** 10
TRAINING_STEPS = 10_000_000
NUM_STEPS = BATCH_SIZE

# Multiprocessing
NUM_EVAL_ENVS = 4
NUM_EVAL_GAMES = 32
NUM_ENVS = 2

# Logging
NUM_REPLAYS = 10
SAVE_FREQ = (TRAINING_STEPS // NUM_REPLAYS) // NUM_ENVS

TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S')

CHECKPOINT_PATH = os.path.join("..", "checkpoints")
MODEL_PATH = os.path.join("..", 'models')
if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)

LOGS_PATH = os.path.join("..", "logs")
CALLBACKS_PATH = os.path.join(LOGS_PATH, f"{TIME_STAMP}")

MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, TIME_STAMP)
MODEL_PATH = os.path.join("..", "models", f"{TIME_STAMP}.zip")

SAVED_MODEL_PATH = os.path.join(CHECKPOINT_PATH, ".zip")
