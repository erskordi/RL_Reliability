#!/bin/bash

echo "Train VAE..."
python VAE_dense.py
echo "Train RL agent..."
python rllib_trainer.py --num-cpus 14
