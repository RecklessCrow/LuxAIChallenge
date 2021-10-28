import os
import shutil
from glob import glob


def main_maker():
    dir = "model_drivers"
    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.mkdir(dir)

    if not os.path.exists("models"):
        os.mkdir("models")

    shutil.copyfile(os.path.join("..", "src", "lux_agent.py"), os.path.join(dir, "lux_agent.py"))
    shutil.copyfile(os.path.join("..", "src", "constants.py"), os.path.join(dir, "constants.py"))

    for model_path in glob(os.path.join("models", "*.zip")):
        with open(os.path.join("model_drivers", f"{model_path.split(os.sep)[-1][:-3]}py"), 'w+') as f:
            f.write("from sb3_contrib import MaskablePPO\n")
            f.write("import os\n")
            f.write("model = MaskablePPO.load(os.path.join(\"..\", \"models\", \"{0}\"))\n".format(model_path.split(os.sep)[-1]))
            f.write(open(os.path.join("base_main.py")).read())
