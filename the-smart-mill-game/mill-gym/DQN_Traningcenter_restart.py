import numpy as np
from src.env.MillEnv_Descrete import MillEnviroment

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow import keras


env = MillEnviroment()

states_shape = env.observation_space.shape
actions = env.action_space.n

def build_model(actions):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(1, 26)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Flatten())
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(actions)


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=int(10e+10), window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=int(6e+6), target_model_update=1e-2)
    return dqn


class Metrics(keras.callbacks.Callback):
    def __init__(self, agent):
        keras.callbacks.Callback.__init__
        self.agent = agent
        
    def on_train_begin(self, logs={}):
        self.metrics = {key : [] for key in self.agent.metrics_names}

    def on_step_end(self, episode_step, logs):
        for ordinal, key in enumerate(self.agent.metrics_names, 0):
            self.metrics[key].append(logs.get('metrics')[ordinal])

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./tensorboards/keras/DQN_512", histogram_freq=0, 
    write_graph=True, write_images=True)

cp_callback = keras.callbacks.ModelCheckpoint(filepath="./models/keras/dqn/dqn_512.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)


dqn = build_agent(model, actions)
metrics = Metrics(dqn)
dqn.compile(Adam(learning_rate=0.1), metrics=['mae']) 
dqn = build_agent(model, actions)

dqn.load_weights('./models/keras/dqn/dqn_weights_512.h5f')
dqn.fit(env, nb_steps=int(9.5e+6), visualize=False, verbose=1, callbacks=[metrics, tensorboard_callback, cp_callback])

dqn.save_weights('./models/keras/dqn/dqn_weights_512.h5f', overwrite=True)
model.save("./models/keras/dqn/model_dqn_512")