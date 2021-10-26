import os
from datetime import datetime

# Observation Vector Sizes
NUM_IDENTIFIERS = 3
NUM_GAME_STATES = 12
NUM_RESOURCES = (5 * 4) + (3 * 3 * 4)
OBSERVATION_SHAPE = (NUM_IDENTIFIERS + NUM_GAME_STATES + NUM_RESOURCES,)

# Observation Constants
MAX_RESEARCH = 200.0
MAX_UNIT_COUNT = 30
NUM_STEPS_IN_DAY = 30
NUM_STEPS_IN_NIGHT = 10

# Reward Constants
CITY_REWARD_MODIFIER = 0.1
UNIT_REWARD_MODIFIER = 0.05
FUEL_REWARD_MODIFIER = 0.0001
LEAD_REWARD_MODIFIER = 1

# Hyper Parameters
LEARNING_RATE = 0.001
GAMMA = 0.995
GAE_LAMBDA = 0.95
BATCH_SIZE = 2048
MAX_STEPS = 10_000_000
NUM_STEPS = BATCH_SIZE
SAVE_FREQ = 50_000

# Multiprocessing
NUM_ENVS = 32
NUM_EVAL_ENVS = 4
NUM_EVAL_GAMES = 30

# File paths
TIME_STAMP = datetime.now().strftime('%m-%d_%H-%M-%S')

CHECKPOINT_PATH = os.path.join("..", "checkpoints")
if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)

LOGS_PATH = os.path.join("..", "logs")
CALLBACKS_PATH = os.path.join(LOGS_PATH, f"{TIME_STAMP}")

MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, TIME_STAMP)
MODEL_PATH = os.path.join("..", "models", f"{TIME_STAMP}.zip")
SAVED_MODEL_PATH = os.path.join("..", "models", "10-25_15-54-27.zip")