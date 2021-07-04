import numpy as np
from src.env.MillEnv_Descrete import MillEnviroment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import CheckpointCallback
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/sb/ppo',
                                         name_prefix='ppo_checkpoint_model')

env = MillEnviroment()
vec_env = make_vec_env(lambda: env, n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tensorboards/sb/", device='auto')
model.learn(total_timesteps=int(10e+10), callback=checkpoint_callback)


model.save("./models/sb/ppo/model_ppo_2")

print("____ AGENT LEARNED ___________")