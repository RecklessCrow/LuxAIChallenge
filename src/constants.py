import os
from datetime import datetime
from luxai2021.game.constants import Constants, LuxMatchConfigs_Default

CONFIGS = LuxMatchConfigs_Default

# Observation Vector Sizes
NUM_IDENTIFIERS = 3
NUM_GAME_STATES = 14
UNIT_VECTOR_SIZE = 7
NUM_RESOURCES = (5 * UNIT_VECTOR_SIZE) + (3 * 3 * UNIT_VECTOR_SIZE)
OBSERVATION_SHAPE = (NUM_IDENTIFIERS + NUM_GAME_STATES + NUM_RESOURCES,)
RESOURCE_LIST = [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL, Constants.RESOURCE_TYPES.URANIUM]

# Observation Constants
MAX_RESEARCH = 200.0
MAX_UNIT_COUNT = 30
NUM_STEPS_IN_DAY = 30
NUM_STEPS_IN_NIGHT = 10

# Reward Constants
CITY_REWARD_MODIFIER = 0.5
NEGATIVE_CITY_MODIFIER = 1.25
UNIT_REWARD_MODIFIER = 0.1
NEGATIVE_UNIT_MODIFIER = 1.25
FUEL_DEPOSITED_REWARD_MODIFIER = 0.0001

WOOD_GATHERED_REWARD_MODIFIER = 0.0001
COAL_GATHERED_REWARD_MODIFIER = WOOD_GATHERED_REWARD_MODIFIER * 20
URANIUM_GATHERED_REWARD_MODIFIER = COAL_GATHERED_REWARD_MODIFIER * 20

RESEARCH_REWARD_MODIFIER = 0.0025
COAL_UNLOCKED = 2
URANIUM_UNLOCKED = 2

CITY_STANDING_REWARD_MODIFIER = CITY_REWARD_MODIFIER * 4
GAME_WIN = 1
GAME_LOSS = -GAME_WIN

# Hyper Parameters
LEARNING_RATE = 0.0001
GAMMA = 0.995
GAE_LAMBDA = 0.95
BATCH_SIZE = 4096
TRAINING_STEPS = 10_000_000
NUM_STEPS = BATCH_SIZE

# Multiprocessing
NUM_EVAL_ENVS = 1  # enables multiprocessing
NUM_EVAL_GAMES = 5
NUM_ENVS = 1

# Logging
NUM_REPLAYS = 10
SAVE_FREQ = (TRAINING_STEPS // NUM_REPLAYS) // NUM_ENVS

TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S')

CHECKPOINT_PATH = os.path.join("..", "checkpoints")
if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)

LOGS_PATH = os.path.join("..", "logs")
CALLBACKS_PATH = os.path.join(LOGS_PATH, f"{TIME_STAMP}")

MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, TIME_STAMP)
MODEL_PATH = os.path.join("..", "models", f"{TIME_STAMP}.zip")
SAVED_MODEL_PATH = os.path.join("..", "models", "10-27_03-15-57.zip")