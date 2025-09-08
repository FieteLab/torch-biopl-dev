#!/bin/bash
#SBATCH --job-name=mazes
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:RTXA6000:1
##SBATCH --gres=gpu:GEFORCERTX2080:2

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=fiete
#SBATCH --mem=50G

#SBATCH --output=/om2/vast/evlab/jackking/torch-bioplnn-dev/train/slurm_outputs/output_%j.txt

source ~/.bashrc

###source /om2/vast/evlab/jackking/.bashrc

module load openmind8/cuda/11.7
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

MT_HOME="/om2/vast/evlab/${USER_NAME}/torch-bioplnn-dev"
# run the .bash_profile file from USER_NAME home directory
# . /home/${USER_NAME}/.bash_profile

conda activate bioplnn
echo $(which python)


# python "${MT_HOME}/scripts/solving_mazes.py" \
#   --model-type 1e1ii1a \
#   --dataset correlated_dots \
#   --lr 0.0012 \
#   --max-gradient 5 \
#   --num-samples 345600 \
#   --num-steps 20 \
#   --max-epochs 400 \
#   --batch-size 128 \
#   --num-neuron-subtypes 8,4 \
#   --fc-dim 512 \
#   --out-channels 8 \
#   --neuron-type-nonlinearity ReLU \
#   --inter-neuron-type-nonlinearity ReLU \
#   --inter-neuron-type-spatial-extents 5,5 \
#   --init-weights none \
#   --scheduler onecycle \
#   --pct-start 0.15 \
#   --div-factor 20 \
#   --wandb-project \
#   --correlation 0.25,0.75 \
#   --n-frames 40 \
#   --max-speed 5 \
#   --seed 4

# python "${MT_HOME}/scripts/solving_mazes.py" \
#   --model-type 1e1ii1a \
#   --lr 0.0012 \
#   --max-gradient 5 \
#   --num-samples 345600 \
#   --num-steps 20 \
#   --max-epochs 400 \
#   --batch-size 1024 \
#   --num-neuron-subtypes 8,4 \
#   --fc-dim 512 \
#   --out-channels 8 \
#   --neuron-type-nonlinearity ReLU \
#   --inter-neuron-type-nonlinearity ReLU \
#   --inter-neuron-type-spatial-extents 5,5 \
#   --init-weights none \
#   --scheduler onecycle \
#   --pct-start 0.15 \
#   --div-factor 20 \
#   --wandb-project mazes \
#   --seed 2

#   python "${MT_HOME}/scripts/solving_mazes.py" \
#   --model-type 1e1ii1a \
#   --lr 0.0012 \
#   --max-gradient 5 \
#   --num-samples 345600 \
#   --num-steps 20 \
#   --max-epochs 400 \
#   --batch-size 1024 \
#   --num-neuron-subtypes 8,4 \
#   --fc-dim 512 \
#   --out-channels 8 \
#   --neuron-type-nonlinearity ReLU \
#   --inter-neuron-type-nonlinearity ReLU \
#   --inter-neuron-type-spatial-extents 5,5 \
#   --init-weights none \
#   --scheduler onecycle \
#   --pct-start 0.15 \
#   --div-factor 20 \
#   --wandb-project mazes \
#   --seed 3

python "${MT_HOME}/scripts/solving_mazes.py" \
  --model-type cnn \
  --lr 0.001 \
  --max-gradient 1 \
  --num-steps 20 \
  --max-epochs 3400 \
  --batch-size 1024 \
  --init-weights none \
  --scheduler onecycle \
  --pct-start 0.15 \
  --div-factor 35 \
  --wandb-project \
  --seed 1 \
  --dataset mazes \
  # --correlation 0.75