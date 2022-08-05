# Remaining Useful Life estimation using Reinforcement Learning

This repo demonstrates a novel process for remaining useful life (RUL) estimation using Variational Autoencoders (VAE) and scalable Reinforcement Learning (RL). The dataset used is the `train_FD002.txt` and can be found in the `CMAPSSData` directory.

The proposed process evolves in two steps:
1) Split the data into three subsets i) VAE training, ii) RL agent training, iii) validation
2) Train a VAE using the first subset of the training data. The decoder is then used in lieu of an actual simulator.
3) Train an RL agent using the second training data subset, and the trained decoder as environment with which it interacts. The actions are resembling generated distribution samples from the VAE's encoded layer
4) Evaluate the agent's performance by estimating the RUL of the systems on the third training data subset.

The files provided are:
- `config.py`: Predetermined hyperparameters for the models used.
- `data_prep.py`: data preprocessing step; includes data normalization, actual life estimation, smoothing
- `VAE_dense.py`: VAE training process. 
- `env.py`: The RL environment; inherits from the [OpenAI Gym](https://www.gymlibrary.ml/) module.
- `rllib_trainer.py`: RL training process, works via the [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) module.

For agent evaluation, you need to run the `RLlib-PPO-policy_eval` Jupyter notebook. 

Modules needed for running the experiments:
```
gym
ray[rllib, serve]
numpy
pandas
itertools
tensorflow
requests
scikit-learn
argparse
jupyter
```

Contact [Erotokritos Skordilis](mailto:sge12@miami.edu) for any questions/comments.