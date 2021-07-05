from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from stable_baselines3 import PPO, ppo
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow import keras

import numpy as np
from src.env.MillEnv_Tester import MillEnviroment

class Controller():

    def __init__(self, dqn_path, dqn_model_path, ppo_path="", dqn_deepth=512, dqn_range=5):
      self.testEnv = MillEnviroment()
      loader = AgentLoader(self.testEnv)
      self.dqn_agent = loader.load_dqn_agent(dqn_path, dqn_model_path)
      if ppo_path != "":
        self.ppo_agent = loader.load_ppo_agent(ppo_path)

    def predict_dqn(self, obs):
      return self.dqn_agent.forward(obs)

    def predict_ppo(self, obs):
      return self.ppo_agent.predict(obs)
    
    def validate_predictions_dqn(self, newObs):
      self.testEnv.set_gameboard(newObs)
      result = self.testEnv.step(self.predict_dqn(newObs))
      if result[0] == True:
        print('propper Move')
        self.testEnv.render()
        if result[3] == True:
          self.validate_predictions_dqn(newObs)
          print('kill')
          self.testEnv.render()
      elif result[0] == False:
        print('invalid Move')
        self.testEnv.render()
    
    def validate_predictions_ppo(self, newObs):
        self.testEnv.set_gameboard(newObs)
        result = self.testEnv.step(self.predict_ppo(newObs)[0])
        if result[0] == True:
          print('propper Move')
          self.testEnv.render()
          if result[3] == True:
            self.validate_predictions_ppo(newObs)
            print('kill')
            self.testEnv.render()
        elif result[0] == False:
          print('invalid Move')
          self.testEnv.render()

class AgentLoader():

  def __init__(self, env):
    self.env = env

  def build_model(self, model_path):
      return keras.models.load_model(model_path)

  def build_agent_dqn(self, model, actions):
      policy = BoltzmannQPolicy()
      memory = SequentialMemory(limit=int(10e+10), window_length=1)
      dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=int(6e+6), target_model_update=1e-2)
      return dqn

  def load_dqn_agent(self, file, model_path):
      actions = self.env.action_space.n
      model = self.build_model(model_path)
      dqn = self.build_agent_dqn(model, actions)
      dqn.compile(Adam(lr=1e-3), metrics=['mae'])
      dqn.load_weights(file)
      return dqn

  def load_ppo_agent(self, file):
      ppo = PPO.load(file)
      return ppo








