import numpy as np
from src.env.MillEnv_Descrete import MillEnviroment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


env = MillEnviroment()
vec_env = make_vec_env(lambda: env, n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tensorboards/sb/", device='auto')
model.learn(total_timesteps=int(9.5e+6))


model.save("./models/sb/ppo/model_ppo")

print("____ AGENT LEARNED ___________")