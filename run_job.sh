#!/bin/bash
#SBATCH --job-name=mazes
#SBATCH --time=2-00:00:00
###SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --cpus-per-task=8

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_evelina9
#SBATCH --mem=50G

#SBATCH --output=/home/jackking/torch-bioplnn-dev/train/slurm_outputs/output_%j.txt

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate bioplnn
export WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08

# find the user name
USER_NAME=$(whoami)
HOME="/home/${USER_NAME}/torch-bioplnn-dev"

echo $(which python)

# python "${HOME}/scripts/solving_mazes.py" \
#   --model-type 1e1ii1a \
#   --dataset mazes \
#   --lr 0.0012 \
#   --max-gradient 2 \
#   --max-epochs 500 \
#   --batch-size 128 \
#   --fc-dim 64 \
#   --init-weights none \
#   --scheduler onecycle \
#   --pct-start 0.15 \
#   --div-factor 20 \
#   --wandb-project \
#   --seed 4 \

# python "${HOME}/scripts/solving_mazes.py" \
#   --model-type td+h-bioplnn \
#   --dataset cabc \
#   --lr 0.0001 \
#   --max-gradient 1 \
#   --num-steps 8 \
#   --max-epochs 100 \
#   --batch-size 16 \
#   --fc-dim 512 \
#   --init-weights none \
#   --scheduler onecycle \
#   --pct-start 0.15 \
#   --div-factor 20 \
#   --wandb-project \
#   --seed 1 \
#   --resolution 256 \
#   --num-samples 40000 \
#   --output-area-index -1 \

# Launch DDP training with torchrun on 2 GPUs
# NUM_GPUS=4
# echo "Using $NUM_GPUS GPUs"

# torchrun --standalone --nproc_per_node=${NUM_GPUS} \
#   "${HOME}/scripts/solving_mazes_in_parallel.py" \
#   --model-type cnn \
#   --dataset cabc \
#   --lr 0.0005 \
#   --max-gradient 2 \
#   --num-steps 10 \
#   --max-epochs 50 \
#   --batch-size 128 \
#   --fc-dim 512 \
#   --init-weights none \
#   --scheduler none \
#   --wandb-project \
#   --seed 4 \
#   --resolution 128 \
  

# python "${HOME}/scripts/solving_mazes.py" \
#   --model-type cnn \
#   --dataset cabc \
#   --lr 0.0005 \
#   --max-gradient 5 \
#   --max-epochs 50 \
#   --batch-size 128 \
#   --fc-dim 1024 \
#   --init-weights none \
#   --scheduler onecycle \
#   --pct-start 0.15 \
#   --div-factor 20 \
#   --wandb-project \
#   --seed 4 \
#   --conv1-out 64 \
#   --conv2-out 128 \
#   --conv3-out 256 \
#   --resolution 128 \

python "${HOME}/scripts/solving_mazes.py" \
  --model-type 1e1ii1a \
  --dataset correlated_dots \
  --lr 0.0012 \
  --max-gradient 2 \
  --max-epochs 400 \
  --batch-size 64 \
  --fc-dim 32 \
  --init-weights none \
  --scheduler onecycle \
  --pct-start 0.15 \
  --div-factor 20 \
  --wandb-project \
  --seed 4 \
  --num-neuron-subtypes 4,4 \
  --num-steps 20 \
  --resolution 128 \
  --directions 0,1