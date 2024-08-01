import numpy as np

channel = [373]#373#398 # abacus # 373 # imagenette: 0, 217, 482, 491, 497, 566, 569, 571, 574, 701
channel = []
channel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
           11, 25, 26, 27, 29, 33, 71, 96, 97, 101,
           128, 130, 131, 134, 136, 256, 257, 258, 259, 260,
           263, 268, 288, 291, 352, 384, 385, 386, 387, 388,
           389, 390, 391, 395, 403, 452, 512, 513, 514, 515,
           516, 517, 518, 519, 520, 521, 522, 523, 524, 525,
           526, 527, 528, 529, 530, 531, 532, 533, 534, 541,
           544, 550, 640, 641, 642, 643, 644, 645, 646, 649,
           652, 653, 657, 672, 768, 772, 776, 777, 779, 805,
           898, 899, 900, 904, 905, 906, 912, 913, 917, 929]
channel = [999]  # testing

# base_params = "--net vgg16 --layer .classifier.Linear6 --optim CholCMA --reps 10"
# base_params = "--net alexnet --layer .classifier.Linear6 --optim CholCMA --reps 10"
# base_params = "--net alexnet-eco-080 --layer .classifier.Linear6 --optim CholCMA --reps 10"
base_params = "--net resnet50 --layer .Linearfc --optim CholCMA --reps 5"
# base_params = "--net resnet50_linf0.5 --layer .Linearfc --optim CholCMA --reps 10"
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
strengths = np.arange(0.5, 1.5, 0.5)
strengths = np.concatenate((strengths, -strengths))
# Now add the perturbed exc/inh
for perturbation_strength in strengths:
    full_line = " ".join([base_params, channels, fc6_gan, perturbation + ("%0.2f" % perturbation_strength)])
    line_list.append(full_line)
    full_line = " ".join([base_params, channels, big_gan, perturbation + ("%0.2f" % perturbation_strength)])
    line_list.append(full_line)

# # Now perturb the absolute weights # not needed for revision
# perturbation = "--perturb kill_topFraction_abs_in_weight_"
# for perturbation_strength in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
#     full_line = " ".join([base_params, channels, fc6_gan, perturbation + ("%0.2f" % perturbation_strength)])
#     line_list.append(full_line)
#     full_line = " ".join([base_params, channels, big_gan, perturbation + ("%0.2f" % perturbation_strength)])
#     line_list.append(full_line)
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