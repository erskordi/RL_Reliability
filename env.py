import bisect
import gym
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape
from sklearn.metrics import mean_squared_error

from data_prep import DataPrep, Vec2Img
from VAE_dense import *


class CMAPSSEnv(gym.Env):
    def __init__(self,
                 df,
                 timestep,
                 obs_size=24,
                 engines=100,
                 engine_lives=[],
                 model=None) -> None:
        super().__init__()
        self.obs_size = obs_size

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size,)) # observations + RUL
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,)) # latent x -> based on mu, sigma coming from sampling layer; here we want the policy to retrieve that value

        self.df = df

        self.num_engines = engines
        self.engine_lives = engine_lives
        self.timestep = timestep

        # Load trained models
        self.model = model
        

    def reset(self):

        self.timestep = np.random.randint(sum(self.engine_lives))
        init_state = self.df.iloc[self.timestep,1:].to_numpy()
        #print(f'Initial state: {init_state}, dimensions: {init_state.shape}')
        
        return init_state # returns the very first observation + RUL % (here 1.000)

    def step(self, action):

        done = False
        self.timestep += 1
        

        #mu, sigma, x = encoder.predict()
        new_state = self.model.predict(action)
        reward = self._reward(self.df.iloc[self.timestep,1:], new_state[0])
        
        if self.df['NormTime'].iloc[self.timestep] == float(0.0):
            done = True
        
        return new_state[0], reward, done, {}

    def render(self) -> None:
        pass

    def _reward(self, y_true, y_pred):
        return - mean_squared_error(y_true, y_pred, squared=False)


if __name__ == "__main__":
    
    file_path = "CMAPSSData/train_FD002.txt"
    num_settings = 3
    num_sensors = 21
    num_units = 20
    step = "RL"

    neurons = [64, 32, 16, 8]

    # Data prep
    data = DataPrep(file=file_path,
                    num_settings=num_settings, 
                    num_sensors=num_sensors, 
                    num_units=num_units, 
                    step=step,
                    normalization_type="01")
    
    df = data.ReadData()
    
    # List of engine lifetimes
    engine_lives = df.groupby(df['Unit']).size()
    engine_lives = engine_lives.tolist()
    num_engines = len(engine_lives)

    # Load decoder
    with open('/Users/erotokritosskordilis/git-repos/RL_Reliability/model_decoder.pkl', 'rb') as f:
        decoder = pickle.load(f)

    ##########################################
    env_config = {
        "df": df,
        "timestep": 0,
        "obs_size": num_settings+num_sensors+1,
        "engines": num_engines,
        "engine_lives": engine_lives, 
        "decoder_model": decoder,
    }

    print("env_config: ", env_config)

    env = CMAPSSEnv(**env_config)
    

    total_cost = 0
    

    for _ in range(1):
        
        done = False
        env.reset()
        print(env.timestep)
        cntr = 0
        s = bisect.bisect_left(np.cumsum(engine_lives), env.timestep)
        steps_to_go = abs(env.timestep - np.cumsum(engine_lives[:s+1])[-1])
        current_step = abs(engine_lives[s] - steps_to_go)
        print(f'Current step: {current_step}, \
            System: {s}, System life: {engine_lives[s]}, Steps until failure: {steps_to_go}')
        
        while not done:
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            total_cost += rew
            cntr += 1
            print(cntr, rew, done)