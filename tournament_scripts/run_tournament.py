import os
from glob import glob

import main_maker

agents = [filename for filename in glob(os.path.join("model_drivers", "*.py"))]

command = f"lux-ai-2021 --rankSystem=\"trueskill\" --tournament {' '.join(agents)} --maxtime 100000"
# print(command)
os.system(command)
