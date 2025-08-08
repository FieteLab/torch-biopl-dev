#!/bin/bash
#SBATCH --job-name=mazes
#SBATCH --time=6-00:00:00
###SBATCH --gres=gpu:a100:1
###SBATCH --gres=gpu:RTXA6000:1
#SBATCH --gres=gpu:GEFORCERTX2080:4

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=fiete
#SBATCH --mem=100G

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

python "${MT_HOME}/scripts/solving_mazes.py"

#spatial_1h1a
#spatial_big_1h1a
#spatial_1e1i1a

    # model.rnn_kwargs.area_kwargs.0.inter_neuron_type_spatial_extents=[5,5] \
    # model.rnn_kwargs.area_kwargs.0.neuron_type_nonlinearity=ReLU \

# MODEL="spatial_1h1a"

# python /om2/user/jackking/torch-bioplnn-dev/examples/trainer.py \
#     model=${MODEL} \
#     data=mazes \
#     train.epochs=200 \
#     train.forward_kwargs.num_steps=10 \
#     optimizer.lr=0.001 \
#     wandb.mode=online \
#     wandb.project=package_mazes \
#     +model_type=${MODEL}


# python examples/trainer.py -m model=spatial_1h1a data=mazes +hydra/sweeper/params=mazes_sweep

# # Run the sweep with multiple models
# python "${MT_HOME}/examples/trainer.py" -m \
#     model=spatial_1h1a,spatial_big_1h1a \
#     data=mazes \
#     hydra/launcher=slurm \
#     hydra.launcher.partition=evlab \
#     hydra.launcher.gres=gpu:a100:1 \
#     hydra.launcher.mem=100G \
#     hydra.launcher.array_parallelism=4 \
#     +hydra/sweeper/params=mazes_sweep