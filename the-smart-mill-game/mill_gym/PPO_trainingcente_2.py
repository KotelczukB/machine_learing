import numpy as np
from src.env.MillEnv_Descrete import MillEnviroment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import CheckpointCallback
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='ppo_final')

env = MillEnviroment()
vec_env = make_vec_env(lambda: env, n_envs=8)

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tensorboards/sb/", device='auto')
model.learn(total_timesteps=int(3e+8), callback=[checkpoint_callback])


model.save("./models/sb/ppo/model_ppo_final")

print("____ AGENT LEARNED ___________")