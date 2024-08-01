#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 0:05:00  # Shorter maximum runtime
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=1-2
#SBATCH --mail-user=giordano_ramos@hms.harvard.edu
#SBATCH -o alexnet-eco-topSilencing_evol_%A_%a.out

module load gcc/9.2.0
module load cuda/11.7

conda activate pytorcha
script_path="/home/gr105/Code/circuit-dissection/insilico_experiments/TopSilencing_Evol_cmp_O2_cluster.py"

cd /home/gr105/Code/circuit-dissection

# Generate the list of arguments using your Python script and store them in an array
mapfile -t ARGS_ARRAY < <(python cluster_scripts/generate_silencing_script.py | sed 's/[][]//g;s/, /\n/g')

# Get the argument corresponding to the current SLURM_ARRAY_TASK_ID
arguments="${ARGS_ARRAY[$SLURM_ARRAY_TASK_ID-1]}"

# Remove surrounding quotes from the argument
temp=${arguments%\'}
temp=${temp#\'}

# Run the script with the arguments
python $script_path $temp
