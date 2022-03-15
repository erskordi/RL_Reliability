import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape
from sklearn.metrics import mean_squared_error

from data_prep import DataPrep, Vec2Img


class CMAPSSEnv(gym.Env):
    def __init__(self,
                 df,
                 timestep,
                 obs_size=24,
                 engines=100,
                 engine_lives=[]) -> None:
        super().__init__()
        self.obs_size = obs_size

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size+1,self.obs_size+1,1)) # observations + RUL
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)) # latent x -> based on mu, sigma coming from sampling layer; here we want the policy to retrieve that value

        self.df = df

        self.num_engines = engines
        self.engine_lives = engine_lives
        self.timestep = timestep

        # Load trained models
        self.encoder, self.decoder = self._call_models()

    def reset(self):
        init_state = self.df.iloc[self.timestep,1:]
        
        return init_state # returns the very first observation

    def step(self, action):

        done = False

        self.timestep += 1

        #mu, sigma, x = encoder.predict()
        new_state = self.df.iloc[self.timestep,1:]
        reconstruction = self.decoder.predict(action)
        reward = self._reward(new_state.to_numpy(), reconstruction[0])
        
        if self.timestep == np.sum(self.engine_lives):
            done = True
        
        
        return new_state, reward, done, {}

    def render(self) -> None:
        pass

    def _call_models(self):
        return (tf.keras.models.load_model('saved_models/encoder'), tf.keras.models.load_model('saved_models/decoder'))

    def _reward(self, y_true, y_pred):
        return -mean_squared_error(y_true, y_pred, squared=False)


if __name__ == "__main__":
    
    file_path = "CMAPSSData/train_FD002.txt"
    num_settings = 3
    num_sensors = 21
    num_units = 100
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


    ##########################################
    env = CMAPSSEnv(df=df, obs_size=num_settings+num_settings+1, timestep=0, engines=num_engines, engine_lives=engine_lives)
    env.reset()

    total_cost = 0
    done = False

    while not done:
        cntr = 0
        for eng in range(num_engines):
            for t in range(engine_lives[eng]):
                action = env.action_space.sample()
                obs, rew, done, _ = env.step(action)
                cntr += engine_lives[eng]
                total_cost += rew
                print(rew, done)