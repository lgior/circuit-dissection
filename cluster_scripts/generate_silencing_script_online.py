import numpy as np

channel = [373]#373#398 # abacus # 373 # imagenette: 0, 217, 482, 491, 497, 566, 569, 571, 574, 701
channel = [373] + [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
channel = [0]  # for single neuron
base_params = "--net alexnet-single-neuron_Caos-12192023-005 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-12192023-008 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-12212023-008 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-01162024-010 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-01172024-006 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-01182024-005 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-01252024-009 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02082024-005 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02092024-006 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02132024-007 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02152024-006 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02202024-007 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02212024-005 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02222024-005 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02272024-005 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-02292024-007 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03052024-005 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03082024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03112024-003 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03122024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03132024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03142024-003 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03182024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03202024-003 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03212024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03222024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03252024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03262024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-03292024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04012024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04022024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04052024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04092024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04102024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04112024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-12042024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04162024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04172024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-19042024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-22042024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-24042024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-25042024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-04302024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-07052024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-05092024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-05132024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-05142024-003 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-16052024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-06062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5" # bad data ignore
base_params = "--net alexnet-single-neuron_Diablito-07062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-17062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-18062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-19062024-003 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-20062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-21062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-24062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-25062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-26062024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-06272024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-08072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-09072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-11072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-12072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-15072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-16072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-17072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-18072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-19072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-22072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Caos-07232024-006 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-24072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-25072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-26072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-29072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"
base_params = "--net alexnet-single-neuron_Diablito-30072024-002 --layer .classifier.Linear6 --optim CholCMA --reps 5"

# base_params = "--net vgg16 --layer .classifier.Linear6 --optim CholCMA --reps 10"
# base_params = "--net alexnet --layer .classifier.Linear6 --optim CholCMA --reps 10"
# base_params = "--net alexnet-eco-080 --layer .classifier.Linear6 --optim CholCMA --reps 10"
# base_params = "--net resnet50 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf0.5 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf1 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf2 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf4 --layer .Linearfc --optim CholCMA --reps 10"
# base_params = "--net resnet50_linf8 --layer .Linearfc --optim CholCMA --reps 10"
fc6_gan = "--G fc6"
# big_gan = "--G BigGAN"
# channels = " ".join(["--chans", str(channel), str(channel + 1)])
channels = " ".join(["--chans", " ".join(map(str, channel))])

# strengths = [-0.2, -0.3, -0.4, -0.5, -0.8]
# strengths = [0.4, 0.5, -0.7, 0.9, -0.9, -1.0]

# initialize list containing jobs to be run
line_list = []
# First run original unit
full_line = " ".join([base_params, channels, fc6_gan])
line_list.append(full_line)
# full_line = " ".join([base_params, channels, big_gan])
# line_list.append(full_line)

perturbation = "--perturb kill_topFraction_in_weight_"
strengths = np.array([0.25, 0.5, 0.75, 1.0])
strengths = np.concatenate((strengths, -strengths))
# Now add the perturbed exc/inh
for perturbation_strength in strengths:
    full_line = " ".join([base_params, channels, fc6_gan, perturbation + ("%0.2f" % perturbation_strength)])
    line_list.append(full_line)
    # full_line = " ".join([base_params, channels, big_gan, perturbation + ("%0.2f" % perturbation_strength)])
    # line_list.append(full_line)

# Now perturb the absolute weights
perturbation = "--perturb kill_topFraction_abs_in_weight_"
for perturbation_strength in [0.25, 0.50, 0.75, 0.99]:
    full_line = " ".join([base_params, channels, fc6_gan, perturbation + ("%0.2f" % perturbation_strength)])
    line_list.append(full_line)
    # full_line = " ".join([base_params, channels, big_gan, perturbation + ("%0.2f" % perturbation_strength)])
    # line_list.append(full_line)
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