#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 0:05:00  # Shorter maximum runtime
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=1-2
#SBATCH --mail-user=giordano_ramos@hms.harvard.edu
#SBATCH -o alexnet-eco-topSilencing_evol_%A_%a.out

conda activate torcha
script_path="M:/Code/Neuro-ActMax-GAN-comparison/insilico_experiments/TopSilencing_Evol_cmp_O2_cluster.py"

cd M:/Code/Neuro-ActMax-GAN-comparison

# Generate the list of arguments using your Python script and store them in an array
mapfile -t ARGS_ARRAY < <(python cluster_scripts/generate_silencing_script.py | sed 's/[][]//g;s/, /\n/g')

# Get the argument corresponding to the current SLURM_ARRAY_TASK_ID
arguments="${ARGS_ARRAY[$SLURM_ARRAY_TASK_ID-1]}"

# Remove surrounding quotes from the argument
temp=${arguments%\'}
temp=${temp#\'}

# Run the script with the arguments
python $script_path $temp
