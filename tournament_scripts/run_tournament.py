from main_maker import *

# main_maker()

agents = []
for filename in glob(os.path.join("model_drivers", "*.py")):
    if "lux" not in filename and "const" not in filename:
        agents.append(filename)

command = f"lux-ai-2021 --rankSystem=\"trueskill\" --tournament {' '.join(agents)} --maxtime 100000 --maxConcurrentMatches 6"
# print(command)
os.system(command)
