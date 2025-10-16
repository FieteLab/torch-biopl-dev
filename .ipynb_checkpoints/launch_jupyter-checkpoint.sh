#!/bin/bash -l
#SBATCH -J jupyter
#SBATCH --time=1-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
###SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
###SBATCH --gres=gpu:1
###SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --mem 300G
###SBATCH --partition=evlab
#SBATCH -o jupyter.out

module load openmind8/gcc/12.2.0
module load openmind8/cuda/12.4
source /om2/user/jackking/miniconda3/etc/profile.d/conda.sh

conda activate bioplnn_2.0

unset XDG_RUNTIME_DIR

PORT=8011

python -m jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --NotebookApp.allow_origin='*' --NotebookApp.port_retries=0
