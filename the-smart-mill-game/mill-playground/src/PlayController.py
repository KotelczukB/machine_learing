from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from stable_baselines3 import PPO
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory



class Controller():

    def __init__(self, playgoundEnv, dqn_agent, ppo_agent):
      self.dqn_agent = dqn_agent
      self.ppo_agent = ppo_agent
      self.testEnv = playgoundEnv

    def predict_dqn(self, obs):
      return self.dqn_agent.forward(obs)

    def predict_ppo(self, obs):
      return self.ppo_agent.forward(obs)
    
    def validate_predictions_dqn(self):
      result = self.testEnv.step(self.predict_dqn())
      if result[1] == True:
        self.testEnv.render()
        if self.testEnv.killMode:
          self.validate_predictions()
      elif result[1] == False:
        self.validate_predictions()
    
    def validate_predictions_ppo(self):
        result = self.testEnv.step(self.predict_ppo())
        if result[1] == True:
          self.testEnv.render()
        if self.testEnv.killMode:
          self.validate_predictions()
        elif result[1] == False:
          self.validate_predictions()


class AgentLoader():

  def __init__(self, env):
    self.env = env

  def build_model(self, actions):
      model = Sequential()
      model.add(Dense(512, activation='relu', input_shape=(1, 26)))
      model.add(Dense(512, activation='relu'))
      model.add(Dense(512, activation='relu'))
      model.add(Flatten())
      model.add(Dense(actions, activation='linear'))
      return model

  def build_agent_dqn(self, model, actions):
      policy = BoltzmannQPolicy()
      memory = SequentialMemory(limit=int(10e+10), window_length=1)
      dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=int(6e+6), target_model_update=1e-2)
      return dqn

  def load_dqn_agent(self, file):
      actions = self.env.action_space.n
      model = self.build_model(actions)
      dqn = self.build_agent_dqn(model, actions)
      dqn.compile(Adam(lr=1e-3), metrics=['mae'])
      dqn.load_weights(file)
      return dqn
  


  def load_ppo_agent(self, file):
      ppo = PPO.load("./model/nine_mens_mill_ppo")
      return ppo








