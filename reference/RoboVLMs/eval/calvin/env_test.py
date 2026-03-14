import os

# BASE_DIR is the path to the RoboVLMs directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#print("BASE_DIR:", BASE_DIR)
path = os.path.join(BASE_DIR, "calvin/dataset/calvin_debug_dataset/validation")
path = "/remote-home/jinminghao/structvla/reference/RoboVLMs/calvin/dataset/calvin_debug_dataset/validation"
from calvin_env.envs.play_table_env import get_env
env = get_env(path, show_gui=False)
print(env.get_obs())
