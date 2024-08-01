#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=1-2
#SBATCH --mail-user=giordano_ramos@hms.harvard.edu
#SBATCH -o alexnet-eco-topSilencing_evol_%j.out

#echo "$SLURM_ARRAY_TASK_ID"
#
#param_list=\
#'--net alexnet-eco-080 --layer .classifier.Linear6 --G fc6 --optim CholCMA --chans 131 132 --reps 10 --perturb kill_topFraction_in_weight_-0.5
#--net alexnet-eco-080 --layer .classifier.Linear6 --G BigGAN --optim CholCMA --chans 131 132 --reps 10 --perturb kill_topFraction_in_weight_-0.5
#'
#
#
#export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
#echo "$unit_name"

#module load gcc/6.2.0
#module load cuda/10.2
##module load conda2/4.2.13

##conda init bash
conda activate torcha
script_path="M:\Code\Neuro-ActMax-GAN-comparison\insilico_experiments\TopSilencing_Evol_cmp_O2_cluster.py"

#cd ~/Github/Neuro-ActMax-GAN-comparison
cd M:/Code/Neuro-ActMax-GAN-comparison
#python3 insilico_experiments/TopSilencing_Evol_cmp_O2_cluster.py  $unit_name
SCRIPT="insilico_experiments/TopSilencing_Evol_cmp_O2_cluster.py"
# This fails if the elements of list are split by whitespaces
#OUTPUT=($(python3 cluster_scripts/generate_silencing_script.py))
#echo $OUTPUT
# This saves the output list to one element per line array, otherwise default is to split by whitespaces.
#https://stackoverflow.com/questions/26162394/convert-a-python-data-list-to-a-bash-array
#mapfile -t ARGS_ARRAY < <(python cluster_scripts/generate_silencing_script.py | sed 's/[][]//g;s/, /\n/g')
mapfile -t ARGS_ARRAY < <(python cluster_scripts/generate_silencing_script_online.py | sed 's/[][]//g;s/, /\n/g')

#for arguments in "${ARGS_ARRAY[@]}"; do
#  temp=${arguments%\'}
#  temp=${temp#\'}
#  echo "python $SCRIPT ${temp} &"; done
for arguments in "${ARGS_ARRAY[@]}";
do
  # escape the "'" surrounding the arguments, otherwise parseargs will fail
#  https://stackoverflow.com/questions/9733338/shell-script-remove-first-and-last-quote-from-a-variable
  temp=${arguments%\'}
  temp=${temp#\'}
#  echo "$SCRIPT" "${temp}"
  python $SCRIPT $temp &
  wait $!
done

#python insilico_experiments/TopSilencing_Evol_cmp_O2_cluster.py --net alexnet-eco-080 --layer .classifier.Linear6 --optim CholCMA --reps 10 --chans 373 374 --G fc6 --perturb kill_topFraction_abs_in_weight_0.99
#python insilico_experiments/TopSilencing_Evol_cmp_O2_cluster.py --net alexnet-eco-080 --layer .classifier.Linear6 --optim CholCMA --reps 4 --chans 373 374 --G BigGAN --perturb kill_topFraction_in_weight_0.1
