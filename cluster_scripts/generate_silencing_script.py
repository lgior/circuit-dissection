import numpy as np

channel = 701#373#398 # abacus # 373 # imagenette: 0, 217, 482, 491, 497, 566, 569, 571, 574, 701
channel = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
# base_params = "--net alexnet --layer .classifier.Linear6 --optim CholCMA --reps 10"
# base_params = "--net alexnet-eco-080 --layer .classifier.Linear6 --optim CholCMA --reps 10"
# base_params = "--net resnet50 --layer .Linearfc --optim CholCMA --reps 10"
base_params = "--net resnet50_linf0.5 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf1 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf2 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf4 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf8 --layer .Linearfc --optim CholCMA --reps 10"
fc6_gan = "--G fc6"
big_gan = "--G BigGAN"
# channels = " ".join(["--chans", str(channel), str(channel + 1)])
channels = " ".join(["--chans", " ".join(map(str, channel))])

# strengths = [-0.2, -0.3, -0.4, -0.5, -0.8]
# strengths = [0.4, 0.5, -0.7, 0.9, -0.9, -1.0]

# initialize list containing jobs to be run
line_list = []
# First run original unit
full_line = " ".join([base_params, channels, fc6_gan])
line_list.append(full_line)
full_line = " ".join([base_params, channels, big_gan])
line_list.append(full_line)

perturbation = "--perturb kill_topFraction_in_weight_"
strengths = np.arange(0.1, 1.1, 0.1)
strengths = np.concatenate((strengths, -strengths))
# Now add the perturbed exc/inh
for perturbation_strength in strengths:
    full_line = " ".join([base_params, channels, fc6_gan, perturbation + ("%0.2f" % perturbation_strength)])
    line_list.append(full_line)
    full_line = " ".join([base_params, channels, big_gan, perturbation + ("%0.2f" % perturbation_strength)])
    line_list.append(full_line)

# Now perturb the absolute weights
perturbation = "--perturb kill_topFraction_abs_in_weight_"
for perturbation_strength in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    full_line = " ".join([base_params, channels, fc6_gan, perturbation + ("%0.2f" % perturbation_strength)])
    line_list.append(full_line)
    full_line = " ".join([base_params, channels, big_gan, perturbation + ("%0.2f" % perturbation_strength)])
    line_list.append(full_line)
# Print line to be fetched by the bash script
print(line_list)
# --net alexnet-eco-080 --layer .classifier.Linear6 --G fc6 --optim CholCMA --chans 131 132 --reps 10 --perturb kill_topFraction_in_weight_-0.5
# --net alexnet-eco-080 --layer .classifier.Linear6 --G BigGAN --optim CholCMA --chans 131 132 --reps 10 --perturb kill_topFraction_in_weight_-0.5
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "fc6", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'10', "--perturb", "kill_topFraction_abs_in_weight_0.99"])

# f.write("Now the file has more content!")
# f.close()
#
# script_path = "M:\Code\Neuro-ActMax-GAN-comparison\insilico_experiments\TopSilencing_Evol_cmp_O2_cluster.py"
#
# with open("test-script-generation.txt", 'w') as output:
#     for row in values:
#         output.write(str(row) + '\n')