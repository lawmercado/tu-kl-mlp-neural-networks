#!/bin/bash
#SBATCH -J job
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err
#SBATCH --gres=gpu:1
#SBATCH -t 87

echo "Executing on $HOSTNAME"

cd /home/vaitses/Desktop/tmp/tu-kl-mlp-neural-networks
python3 model-select.py

echo "Finished execution"