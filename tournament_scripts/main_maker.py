from glob import glob
import os


dir = "model_drivers"
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

os.chdir("models")
for model_path in glob(os.path.join("*.zip")):
    with open(os.path.join("..", "model_drivers", f"{model_path[:-4]}.py"), 'w+') as f:
        f.write("from stable_baselines3 import PPO\n")
        f.write(f"model = PPO.load(\"{model_path}\")\n")
        f.write(open(os.path.join("..", "base_main.py")).read())