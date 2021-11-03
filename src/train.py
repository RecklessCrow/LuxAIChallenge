
from constants import *
from sb3_model import make_model, make_training_env, make_callbacks


def train():
    """
    Main training loop
    :return:
    """

    train_env = make_training_env()
    model = make_model(env=train_env)
    callbacks = make_callbacks(model)

    print("Training Model...")
    model.learn(
        total_timesteps=TRAINING_STEPS,
        callback=callbacks
    )

    print("Saving Model...")
    if not os.path.exists(MODEL_PATH):
        model.save(path=MODEL_PATH)

    print("Done training model.")
