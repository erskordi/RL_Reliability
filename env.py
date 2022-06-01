import bisect
import gym
import numpy as np
import pickle
import requests
import ray
import tensorflow as tf

from ray import serve
from tensorflow.python.ops.gen_array_ops import shape
from sklearn.metrics import mean_squared_error

from data_prep import DataPrep
from tf_serve_models import TFEncoderDecoderModel
from VAE_dense import *


class CMAPSSEnv(gym.Env):
    def __init__(self,
                 df,
                 timestep,
                 obs_size=24,
                 engines=100,
                 engine_lives=[],
                 models=[None, None],
                 env_type="batch") -> None:
        super().__init__()
        self.obs_size = obs_size

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size,)) # observations + RUL
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,)) # latent x -> based on mu, sigma coming from sampling layer; here we want the policy to retrieve that value

        self.df = df

        self.num_engines = engines
        self.engine_lives = engine_lives
        self.timestep = timestep

        # Load trained models
        self.models = models
        

    def reset(self):

        self.timestep = np.random.randint(sum(self.engine_lives))
        init_state = self.df.iloc[self.timestep,1:].to_numpy()
        #print(f'Initial state: {init_state}, dimensions: {init_state.shape}')
        
        return init_state 

    def step(self, action):

        done = False

        """ Actual observation (from data) """
        actual_obs = self.df.iloc[self.timestep,1:].tolist()
                
        actuals = requests.get(
            "http://localhost:8000/saved_models", 
            json={"array": [[actual_obs], action.tolist()]
                } #np.random.uniform(0,1,1).tolist()
            )

        """ Actual latent x from encoder """
        actual_latent_state = actuals.json()['predictions'][0][0]

        """ New state given action """
        new_state = actuals.json()['predictions'][1][0]

        reconstructed = requests.get(
            "http://localhost:8000/saved_models", 
            json={"array": [[actual_obs], actual_latent_state]
                } #np.random.uniform(0,1,1).tolist()
            )

        """ Reconstructed state given true X"""    
        _ = reconstructed.json()['predictions'][0][0]
        reconstructed_state = reconstructed.json()['predictions'][1][0]

        reward = self._reward(new_state, reconstructed_state)
        
        self.timestep += 1

        if self.df['NormTime'].iloc[self.timestep] == float(0.0):
            done = True

        return new_state, reward, done, {}

    def render(self) -> None:
        env_snapshot = {
            "Current observation": self.df.iloc[self.timestep,1:].tolist(),
            "New state": list(self.decoder_model.predict(action, verbose=0)[0]),
        }

    def _reward(self, est_state, rec_state):
        return -mean_squared_error(est_state, rec_state, squared=False)


if __name__ == "__main__":
    serve.start()
    TFEncoderDecoderModel.deploy(['./saved_models/encoder','./saved_models/decoder'])
    
    const = Config()
    neurons = const.VAE_neurons

    # Data prep
    data = DataPrep(file = const.file_path,
                    num_settings = const.num_settings,
                    num_sensors = const.num_sensors,
                    num_units = const.num_units[1],
                    prev_step_units = const.prev_step_units[1],
                    step = const.step[1],
                    normalization_type="01")

    df = data.ReadData()
    
    # List of engine lifetimes
    engine_lives = df.groupby(df['Unit']).size()
    engine_lives = engine_lives.tolist()
    num_engines = len(engine_lives)

    # Environment types
    env_types = ["batch", "intertemporal"]

    ##########################################
    env_config = {
        "df": df,
        "timestep": 0,
        "obs_size": const.num_settings+const.num_sensors+1,
        "engines": num_engines,
        "engine_lives": engine_lives, 
        "models": [None, None],
        "env_type": env_types[1],
    }

    print("env_config: ", env_config)

    env = CMAPSSEnv(**env_config)

    total_cost = 0
    

    for _ in range(1):
        
        init_state = env.reset()
        cntr = 0

        """"""
        s = bisect.bisect_left(np.cumsum(engine_lives), env.timestep)
        steps_to_go = abs(env.timestep - np.cumsum(engine_lives[:s+1])[-1]) - 1
        current_step = abs(engine_lives[s] - steps_to_go)
        print(f'Current step: {current_step},\
            System: {s}, System life: {engine_lives[s]}, Steps until failure: {steps_to_go}')
        

        while True:
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            total_cost += rew
            cntr += 1
            print(rew, done)
            if done:
                break
        print(total_cost)