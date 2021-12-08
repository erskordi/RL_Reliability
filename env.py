import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape

from data_prep import Vec2Img


class CMAPSSEnv(gym.Env):
    def __init__(self,
                 obs_size=24) -> None:
        super().__init__()
        self.obs_size = obs_size

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.obs_size,self.obs_size,1))

        self.action_space = gym.spaces.Tuple((
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)), # mean
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)), # std
            gym.spaces.Box(0, 1, shape=(1,))) # RUL
        )

    def reset(self):
        pass

    def step(self, action):

        # Load trained decoder to use as environment
        new_model = tf.keras.models.load_model('saved_models/environment')

    def render(self) -> None:
        pass


if __name__ == "__main__":
    pass