""" Running analysis on yolov7 environment to extract object detections"""
import re
import sys
import os
if os.environ['COMPUTERNAME'] == 'MNB-PONC-D21184':
    sys.path.append(r"M:\Code\Neuro-ActMax-GAN-comparison")  # new PC
else:
    sys.path.append(r"\C:\Users\giordano\Documents\Code\Neuro-ActMax-GAN-comparison")  #oldPC
import tqdm
import numpy as np
from os.path import join
import matplotlib as mpl
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names, save_as_images) # install it from huggingface GR
from core.utils.CNN_scorers import TorchScorer
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square
from core.utils.layer_hook_utils import get_module_names, layername_dict, register_hook_by_module_names
from core.utils.Optimizers import CholeskyCMAES, HessCMAES, ZOHA_Sphere_lr_euclid
from core.utils.plot_utils import saveallforms, save_imgrid
from core.utils.perturbation_utils import apply_silence_weight_fraction_fully_connected_unit as kill_fc_top_in_fraction
from core.utils.perturbation_utils import apply_silence_weight_topn_fully_connected_unit as kill_fc_top_in_n

import pandas as pd
from insilico_experiments import analyze_evolutions as anevo
import matplotlib as mpl
import itertools
import seaborn as sns
import pickle
from functools import reduce
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from tqdm import tqdm


"""with a correct cmaes or initialization, BigGAN can match FC6 activation."""
# Folder to save
if os.environ['COMPUTERNAME'] == 'MNB-PONC-D21184':  # new pc
    rootdir = r"M:\Data"
    rootdir = r"C:\Users\gio\Data"  # personal folder gets full at 50GB
else:
    rootdir = r"C:\Users\giordano\Documents\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"

figures_dir = os.path.join('C:', 'Users', 'gio', 'Data', 'figures', 'silencing')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
#%%
yolo_dir = r'C:\Users\gio\Documents\code\yolov7'
sys.path.append(yolo_dir)
from hubconf import custom


model = custom(path_or_model=os.path.join(yolo_dir, 'yolov7.pt'))

#%%
import cv2
from numpy import random
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_img_size
from utils.datasets import letterbox
from utils.plots import plot_one_box

imgsz = 640
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names] # this is just a random color for each class


def get_image_objectness(tmp_tensor, model, stride=32):

    save_txt = True
    save_conf = True
    save_img = True


    tmp_images = []
    resized_images = []
    # convert each element of the first dimension of the tensor to a cv2 image, new image variable in RGB format
    for i in range(tmp_tensor.shape[0]):
        tmp_image = tmp_tensor[i].permute(1, 2, 0).numpy()
        tmp_images.append(tmp_image)
        resized_images.append(
            np.ascontiguousarray(letterbox(tmp_image, new_shape=640, stride=stride)[0])
        )

    # stack the image list into a tensor
    resized_tensor = torch.stack([torch.from_numpy(img) for img in resized_images]).permute(0, 3, 1, 2)

    with torch.no_grad():
        output_resized = model(resized_tensor)
    # output is a tuple with 2 elements, element 0 is a tensor 20x4032x85, element 1 is a list of 3 tensors,
    # first list element is a tensor 20x3x32x32x85, second list element is 20x3x16x16x85 and third 20x3x8x8x85
    # 20 is the batch size, [n_images, n_anchor_sizes, n_grid_rows, n_grid_cols, n_features]
    # n_anchor_sizes - The anchor sizes associated with the head (usually 3).
    # n_grid_rows - The number of grid rows in the feature map (usually 32, 16, and 8).
    # n_grid_cols - The number of grid columns in the feature map (usually 32, 16, and 8).
    # n_features - The number of features in each grid cell (usually 85).
    # n_features = 4 + 1 + n_classes, where 4 are the bounding box coordinates,
    # 1 is the objectness score and n_classes is the number of classes

    predictions_resized = non_max_suppression(output_resized[0], conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)



    for i_img, img_detection in enumerate(predictions_resized):

        im0 = tmp_images[i_img] * 255
        im0 = np.ascontiguousarray(im0, dtype=np.uint8)
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        img = resized_tensor[i_img].unsqueeze(0)

        if img_detection.nelement() == 0:
            print(f"Image {i_img + 1}: No detections")
            continue
        # Rescale boxes from img_size to im0 size
        img_detection[:, :4] = scale_coords(img.shape[2:], img_detection[:, :4], im0.shape).round()

        s = ''
        for i_class in img_detection[:, -1].unique():
            # Print results
            n = (img_detection[:, -1] == i_class).sum()  # detections per class
            s += f"{n} {names[int(i_class)]}{'s' * (n > 1)}, "  # add to string
        print(f"Image {i_img + 1}: '{s}'")

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # Write results
        for *xyxy, conf, cls in reversed(img_detection):
            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                print(line)
            if save_img:  # Add bbox to image
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_box = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                # display results of plot_one_box that used cv2.rectangle
                cv2.imshow('plot_one_box', im0)
                cv2.waitKey(100)

    # extract the fifth element of every list in the list of predictions
    # this is a scalar with the confidence of the prediction, this needs two nested loops one over predictions
    # and one over the elements of the list, predictions is a list of lists, some may be empty
    # confidences = [pred[:, 4].detach().cpu().numpy() if pred.nelement() > 0 else np.zeros(1) for pred in predictions]
    confidences_resized = [pred[:, 4].detach().cpu().numpy() if pred.nelement() > 0 else np.zeros(1) for pred in predictions_resized]

    n_objects = [len(pred) for pred in predictions_resized]
    mean_objectness = [np.mean(conf) for conf in confidences_resized]
    max_objectness = [np.max(conf) for conf in confidences_resized]
    return mean_objectness, max_objectness, n_objects

#%%

figures_dir = os.path.join('C:', 'Users', 'gio', 'Data', 'figures', 'silencing')
os.makedirs(figures_dir, exist_ok=True)

# TODO refactor following repeated code into another function or method
rootdir = r"C:\Users\gio\Data"  # since my home folder got full

# check if a folder named preprocessed_data exists, if not create it
preprocessed_data_dir_path = join(rootdir, 'preprocessed_data')
if not os.path.exists(preprocessed_data_dir_path):
    os.makedirs(preprocessed_data_dir_path)

# net_str = 'vgg16'
# net_str = 'alexnet'  # 'resnet50', 'alexnet-eco-080'
# net_str = 'resnet50'  # 'resnet50', 'alexnet-eco-080'
net_str = 'resnet50_linf2'
net_str = 'alexnet-single-neuron2'

if any(s in net_str for s in ['alexnet-eco', 'resnet50']):
    layer_str = '.Linearfc'  # for resnet50 and alexnet-eco-080
else:
    layer_str = '.classifier.Linear6'  # for alexnet

# print message showing we are analyzing the following network and layer
print(f'Analyzing network {net_str} and layer {layer_str}')
print(f'Possible units to analyze: {anevo.get_recorded_unit_indices(rootdir, net_str, layer_str)}')

perturbation_regex = r'.*(_kill.*)$'
perturbation_pattern = '_kill_topFraction_'

full_experiment_suffixes = anevo.get_complete_experiment_list()
silencing_experiment_suffixes = []
for suffix in full_experiment_suffixes:
    if 'kill' in suffix:
        silencing_experiment_suffixes.extend(full_experiment_suffixes[suffix])


complete_unit_indices = anevo.get_complete_experiment_units(
    rootdir, net_str, layer_str, perturbation_pattern, perturbation_regex, silencing_experiment_suffixes
)
complete_unit_index_list = [key for key, value in complete_unit_indices.items() if value is None]

imagenette_units = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

assert set(imagenette_units + [373]).issubset(set(complete_unit_index_list))

#%%

# define this "M:\Documents\Training-data-metadata\imagenet_classes.txt" in an os agnostic way
imagenet_metadata_path = join('M:\\', 'Documents', 'Training-data-metadata', 'imagenet_classes.txt')
with open(imagenet_metadata_path, 'r') as f:
    imagenet_index2class = f.readlines()

imagenet_index2class = [line.strip() for line in imagenet_index2class]



#%%
# load dataframes and images from file
file_name = f'{net_str}_{layer_str}_dataframes_dict.pkl'
with open(join(preprocessed_data_dir_path, file_name), 'rb') as f:
    df_dict = pickle.load(f)

#%%

file_name = f'{net_str}_{layer_str}_images_dict.pkl'
with open(join(preprocessed_data_dir_path, file_name), 'rb') as f:
    image_dict = pickle.load(f)


#%%
compute_objectness = False

if compute_objectness:
    # get objectness for each list of images in the image_dict
    #initialize the dictionary, it is nested
    objectness_dict = {}
    for key in image_dict.keys():
        objectness_dict[key] = {}
        objectness_dict[key]['mean_objectness'] = []
        objectness_dict[key]['max_objectness'] = []
        objectness_dict[key]['n_objects'] = []

    for i, (key, unit_tensor_list) in enumerate(image_dict.items()):
        print(f"Processing unit {i + 1} of {len(image_dict)}")
        mean_objectness = []
        max_objectness = []
        n_objects = []
        for j, condition_img_tensor in enumerate(unit_tensor_list):
            print(f"Processing condition {j + 1} of {len(unit_tensor_list)}")
            tmp_objectness = []
            tmp_max_objectness = []
            tmp_n_objects = []
            # call get_image_objectness for each list of images
            tmp_objectness, tmp_max_objectness, tmp_n_objects = get_image_objectness(condition_img_tensor, model, stride=32)
            mean_objectness.append(tmp_objectness)
            max_objectness.append(tmp_max_objectness)
            n_objects.append(tmp_n_objects)
        objectness_dict[key]['mean_objectness'] = mean_objectness
        objectness_dict[key]['max_objectness'] = max_objectness
        objectness_dict[key]['n_objects'] = n_objects

    # make the objectness dict into a dict of dataframes
    objectness_df_dict = {}
    for key in objectness_dict.keys():
        objectness_df_dict[key] = pd.DataFrame(objectness_dict[key])

    # merge the objectness dataframes with the df_dict dataframes in a new dict
    df_score_objectness_dict = {}
    for key in objectness_df_dict.keys():
        df_score_objectness_dict[key] = pd.merge(df_dict[key], objectness_df_dict[key], left_index=True, right_index=True)

    # explode the columns of lists into rows
    for key in df_score_objectness_dict.keys():
        # find the columns that are lists
        list_bool_df = (df_score_objectness_dict[key].map(type) == list).all()
        list_columns = list_bool_df[list_bool_df].index.tolist()
        df_score_objectness_dict[key] = df_score_objectness_dict[key].explode(list_columns)

    file_name = f'{net_str}_{layer_str}_dataframes_dict_objectness.pickle'
    # save the df_dict as a pickle
    with open(join(preprocessed_data_dir_path, file_name), 'wb') as handle:
        pickle.dump(df_score_objectness_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
dddfff
#%%
net_str_list = ['alexnet', 'vgg16',
                'resnet50', 'resnet50_linf0.5', 'resnet50_linf1', 'resnet50_linf2', 'resnet50_linf4', 'resnet50_linf8']
# net_str_list = ['alexnet-single-neuron2']

layer_str_list = ['.Linearfc' if any(s in net_str for s in ['alexnet-eco', 'resnet50']) else '.classifier.Linear6'
                  for net_str in net_str_list]
#%%
df_mean_objectness_per_condition = pd.DataFrame()

for net_str, layer_str in zip(net_str_list, layer_str_list):
    # load the df_dict from a pickle
    file_name = f'{net_str}_{layer_str}_dataframes_dict_objectness.pickle'
    with open(join(preprocessed_data_dir_path, file_name), 'rb') as handle:
        df_score_objectness_dict = pickle.load(handle)

    # add a new column called total_objectness that is the product of the objectness and the number of objects
    for key in df_score_objectness_dict.keys():
        df_score_objectness_dict[key]['total_objectness'] = \
            df_score_objectness_dict[key]['mean_objectness'] * df_score_objectness_dict[key]['n_objects']

    # combine all dict dataframes into one dataframe
    # loop over the dictionary and take the mean over groupby type and strength

    # initialize the dataframe
    for key in df_score_objectness_dict.keys():
        unit_score = f'scores_{key}'
        tmpdf = df_score_objectness_dict[key].groupby(['type', 'strength'])[
            ['mean_objectness', 'max_objectness', 'n_objects', 'total_objectness']].mean().reset_index()
        tmpdf['unit'] = key
        tmpdf['net'] = net_str
        df_mean_objectness_per_condition = pd.concat([df_mean_objectness_per_condition, tmpdf], ignore_index=False)


    # # save the df_dict as a pickle
    # file_name = f'{net_str}_{layer_str}_mean_objectness_per_condition.pickle'
    # with open(join(preprocessed_data_dir_path, file_name), 'wb') as handle:
    #     pickle.dump(df_mean_objectness_per_condition, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
# combine all dict dataframes into one dataframe
# loop over the dictionary and take the mean over groupby type and strength

# initialize the dataframe
df_mean_objectness_per_condition = pd.DataFrame()
for key in df_score_objectness_dict.keys():
    unit_score = f'scores_{key}'
    tmpdf = df_score_objectness_dict[key].groupby(['type', 'strength'])[
        ['mean_objectness', 'max_objectness', 'n_objects', 'total_objectness']].mean().reset_index()
    # add column with key name
    tmpdf['unit'] = key
    # concatenate the dataframes
    df_mean_objectness_per_condition = pd.concat([df_mean_objectness_per_condition, tmpdf], ignore_index=True)

#%%

# set matplotlib backend to Agg
# mpl.use('TkAgg')
# mpl.interactive(False)
mpl.use('module://backend_interagg')
#%%
# plot the objectness for each unit grouped by type, x axis strength and y axis objectness, each key of the dict is a unit

# make a make an rcParams dict to set temporary rcParams for larger fonts of all text objects in the plot
rcParams = {'font.size': 28,
            'axes.labelsize': 24,
            'axes.titlesize': 26,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 20}


# get the number of keys in dict
n_keys = len(df_score_objectness_dict.keys())
# get the number of rows and columns for the subplots
n_rows = int(np.ceil(np.sqrt(n_keys)))
n_cols = int(np.ceil(n_keys / n_rows))

y_var = 'max_objectness'

with plt.rc_context(rcParams):
    # create the figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(25, 25))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for key in df_score_objectness_dict.keys():
        # plot the objectness for each unit grouped by type, x axis strength and y axis objectness, each key of the dict is a unit
        # chex if axes is a list with more than one element
        if isinstance(axs, np.ndarray):
            ax = axs.flatten()[list(df_score_objectness_dict.keys()).index(key)]
        else:
            ax = axs
        sns.lineplot(data=df_score_objectness_dict[key], x='strength', y=y_var, hue='type', ax=ax, lw=3,
                     hue_order=['abs', 'inh', 'exc', 'none'],
                     palette=['tab:purple', 'tab:orange', 'tab:green', 'tab:blue'],
                     err_kws={'edgecolor': None})
        ax.set_title(f"{key} ({imagenet_index2class[key]}) objectness")
        # legend to the right outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.suptitle(f"{net_str} {layer_str} {y_var}")
    plt.tight_layout()
    plt.show()


#%%
# plot the mean objectness for each unit grouped by type, x axis strength and y axis objectness, one line per unit
# y_var = 'mean_objectness'
y_var = 'max_objectness'
# y_var = 'total_objectness'
# y_var = 'n_objects'

rcParams = {'font.size': 20,
            'axes.labelsize': 18,
            'axes.titlesize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 10}
# import maxNLocator
from matplotlib.ticker import MaxNLocator
with plt.rc_context(rcParams):
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.sca(ax)
    # find all rows with net == net and type == none
    controls = df_mean_objectness_per_condition[df_mean_objectness_per_condition.type == 'none']
    controls = controls[y_var].values
    # then compute the confidence interval
    ci = sns.utils.ci(sns.algorithms.bootstrap(controls, n_boot=1000, func=np.mean), which=95, axis=0)
    # plot the mean and confidence interval, without edgecolor
    plt.fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
    plt.axhline(y=controls.mean(), color='k', linestyle='-', alpha=0.1, lw=3)
    sns.lineplot(data=df_mean_objectness_per_condition, x='strength', y=y_var, hue='type', lw=3,
                 hue_order=['abs', 'inh', 'exc', 'none'],
                 palette=['tab:purple', 'tab:orange', 'tab:green', 'tab:blue'],
                 err_kws={'edgecolor': None})

    # # scatterplot the individual points and jitter them
    # jitter_x = anevo.jitter(df_mean_objectness_per_condition.strength, 0.01)
    # sns.scatterplot(data=df_mean_objectness_per_condition, x=jitter_x, y=y_var, hue='type', alpha=0.5)
    plt.title(f"{y_var} across \n units and networks")
    # legend to the right outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # start y limit at 0
    ax.set_ylim(bottom=0)
    # x limits 0 to 1
    ax.set_xlim(left=0, right=1)
    # set 3 ticks on x axis
    ax.xaxis.set_major_locator(MaxNLocator(3))
    # xticks to 0 0.5 1
    ax.set_xticks([0, 0.5, 1])
    # set 3 ticks on y axis
    ax.yaxis.set_major_locator(MaxNLocator(3))
    # set aspect ratio to be square 1/aspect
    ax.set_aspect(1 / ax.get_data_ratio())
    plt.tight_layout()
    plt.show()

figure_filename = os.path.join(figures_dir, f"yolo_{y_var}_vs_strength_across_units_and_networks")
fig.savefig(figure_filename + '.png', bbox_inches='tight', dpi=300)
fig.savefig(figure_filename + '.pdf', bbox_inches='tight', dpi=300)

#%%
# # make one subplot per net, for each net plot a subplot with max objectness vs strength for each unit as a line,
# with hue by type
y_var = 'max_objectness'
# compute unique number of net
nets = df_mean_objectness_per_condition.net.unique()
n_nets = len(nets)
# get the number of rows and columns for the subplots
n_rows = int(np.ceil(np.sqrt(n_nets)))
n_rows = 2
n_cols = int(np.ceil(n_nets / n_rows))

rcParams = {'font.size': 16,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12}

with plt.rc_context(rcParams):
    # create the figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), sharex=True, sharey=True)
    for net in nets:
        # plot the objectness for each unit grouped by type, x axis strength and y axis objectness, each key of the dict is a unit
        ax = axs.flatten()[list(nets).index(net)]
        plt.sca(ax)
        # find all rows with net == net and type == none
        net_controls = df_mean_objectness_per_condition[(df_mean_objectness_per_condition.net == net) &
                                                            (df_mean_objectness_per_condition.type == 'none')]
        net_controls = net_controls[y_var].values
        # then compute the confidence interval
        ci = sns.utils.ci(sns.algorithms.bootstrap(net_controls, n_boot=1000, func=np.mean), which=95, axis=0)
        # plot the mean and confidence interval, without edgecolor
        plt.fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
        plt.axhline(y=net_controls.mean(), color='k', linestyle='-', alpha=0.1, lw=3)
        sns.lineplot(data=df_mean_objectness_per_condition[df_mean_objectness_per_condition.net == net],
                     x='strength', y=y_var, hue='type', ax=ax, lw=3,
                     hue_order=['abs', 'inh', 'exc', 'none'],
                     palette=['tab:purple', 'tab:orange', 'tab:green', 'tab:blue'],
                     err_kws={'edgecolor': None}
                     )
        ax.set_title(f"{net}")
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelbottom=True)
        # set xticks to 0 0.5 1.0
        ax.set_xticks([0, 0.5, 1.0])
        # set only 3 yticks from current yticks
        ax.set_yticks(np.linspace(0, ax.get_yticks()[-1], 3))
        if list(nets).index(net) == len(nets) - 1:
            # legend to the right outside the plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.get_legend().remove()
        # equal data ratio
        ax.set_aspect(1 / ax.get_data_ratio())
    fig.suptitle(f"{y_var}")
    plt.tight_layout()
    plt.show()


#%%


fig_name = f'yolo_{y_var}_vs_strength_per_net'
fig.savefig(os.path.join(figures_dir, fig_name + '.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(figures_dir, fig_name + '.pdf'), dpi=300, bbox_inches='tight')

#%% ############ From here on is network weights analysis that can be moved to another function############
# import TorchScorer class
from core.utils.CNN_scorers import TorchScorer

network_list = ['alexnet', 'vgg16',
                'resnet50', 'resnet50_linf0.5', 'resnet50_linf1', 'resnet50_linf2', 'resnet50_linf4', 'resnet50_linf8']
# net = 'alexnet'
# make a dictionary with network names as keys with empty lists as values
net_weights_dict = {net: [] for net in network_list}
for inet, net in enumerate(network_list):
    scorer = TorchScorer(net)
    model = scorer.model.eval()
    # get last layer weights
    if any(s in net for s in ['alexnet-eco', 'resnet50']):
        layer_str = 'fc'  # for resnet50 and alexnet-eco-080
        net_weights_dict[net] = scorer.model.__getattr__(layer_str).weight.data.cpu().numpy()
    else:
        layer_str = 'classifier'  # for alexnet
        sequential_index = 6
        net_weights_dict[net] = scorer.model.__getattr__(layer_str)[sequential_index].weight.data.cpu().numpy()
#%%
# get the input weights of the units
imagenette_units = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701] + [373]

# make a dataframe with a column for each network and a row for each unit
weights_df = pd.DataFrame(columns=network_list, index=imagenette_units)

# populate each column with the array of weights of the unit
for inet, net in enumerate(network_list):
    for iunit, unit in enumerate(imagenette_units):
        weights_df.loc[unit, net] = net_weights_dict[net][iunit, :]

# melt the dataframe to convert the arrays into a column
weights_df = weights_df.reset_index().melt(id_vars='index', var_name='network', value_name='weights').explode('weights')

#%%

# boxplot of the weights by network
sns.violinplot(x='network', y='weights', data=weights_df)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# get the input weights of the units
imagenette_units = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701] + [373]

# make a subplot grid with similar number of rows and columns
nrows = int(np.ceil(np.sqrt(len(network_list))))
ncols = int(np.ceil(len(network_list) / nrows))
nrows = 2
ncols = 4


rcParams = {'font.size': 24,
            'axes.labelsize': 20,
            'axes.titlesize': 22,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 14}
with mpl.rc_context(rcParams):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

    # plot the histograms for each key in the dictionary
    for ikey, (key, weights) in enumerate(net_weights_dict.items()):
        ax = axes.flatten()[ikey]
        print(weights.shape)
        # get the columns of the weights matrix that correspond to the units
        unit_weights = weights[imagenette_units, :]
        # plot the histograms by row as kde curves
        for irow in range(unit_weights.shape[0]):
            sns.kdeplot(unit_weights[irow, :], ax=ax, label=f'unit {imagenette_units[irow]}', cumulative=False)
        # plot vertical x line at 0
        ax.axvline(0, color='k', linestyle='--')
        # # set ax to log scale
        # ax.set_xscale('symlog')
        # #  ylog
        # ax.set_yscale('symlog')
        ax.set_xlabel('weight')
        ax.set_ylabel('kde')
        ax.set_title(key)
        # one to one aspect ratio in physical units
        ax.set_aspect(1/ax.get_data_ratio())
        # legend outside to the right centered vertically on the whole figure
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0.)
        # log scale
    plt.tight_layout()

    plt.show()


fig_name = f'weights_per_unit_per_net'
fig.savefig(os.path.join(figures_dir, fig_name + '.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(figures_dir, fig_name + '.pdf'), dpi=300, bbox_inches='tight')

#%%
import statsmodels.api as sm

kde_weights_dict = {}
kde_support_dict = {}


for ikey, (key, weights) in enumerate(net_weights_dict.items()):

    # get min and max weigths
    wmin = np.min(weights)
    wmax = np.max(weights)
    # linspace of min and max of 2048 points
    weights_points = np.linspace(wmin, wmax, weights.shape[1])
    kde_support_dict[key] = weights_points

    kde_array = np.zeros((weights.shape[0], weights_points.shape[0]))
    # get density from statsmodel KDE for amatrix of weights row by row
    for irow in range(weights.shape[0]):
        kde = sm.nonparametric.KDEUnivariate(weights[irow, :]).fit()
        kde_array[irow, :] = kde.evaluate(weights_points)
    kde_weights_dict[key] = kde_array


#%%

# save the kde weights dict and the kde support dict to a pickle file
# combine them into single dict
kde_dict = {'kde_weights_dict': kde_weights_dict,
            'kde_support_dict': kde_support_dict}

# save the dict to a pickle file in preprocessed data dir
with open(os.path.join(preprocessed_data_dir_path, 'kde_dict.pickle'), 'wb') as handle:
    pickle.dump(kde_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
# load the kde dict from the pickle file
with open(os.path.join(preprocessed_data_dir_path, 'kde_dict.pickle'), 'rb') as handle:
    kde_dict = pickle.load(handle)

kde_weights_dict = kde_dict['kde_weights_dict']
kde_support_dict = kde_dict['kde_support_dict']

#%%
# import FormatStrFormatter
from matplotlib.ticker import FormatStrFormatter

# for each network plot the kde of the weights of each unit
nrows = int(np.ceil(np.sqrt(len(network_list))))
ncols = int(np.ceil(len(network_list) / nrows))
nrows = 2
ncols = 4

with mpl.rc_context(rcParams):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))

    # plot the histograms for each key in the dictionary
    for ikey, (key, kde_array) in enumerate(kde_weights_dict.items()):
        ax = axes.flatten()[ikey]
        plt.sca(ax)
        # plot the kde array with x axis as weights_points and y axis as units
        wmin = kde_support_dict[key].min()
        wmax = kde_support_dict[key].max()

        lognorm = mpl.colors.SymLogNorm(linthresh=1e-8, linscale=0.02, vmin=kde_array.min(), vmax=kde_array.max())
        linnorm = mpl.colors.Normalize(vmin=kde_array.min(), vmax=kde_array.max())
        plt.imshow(kde_array, aspect='auto', extent=(wmin, wmax, 0, kde_array.shape[0]),
                   norm=linnorm)
        plt.xlabel('weight')
        plt.ylabel('unit')
        # plot only 3 tick marks per axis
        plt.locator_params(axis='x', nbins=3)
        plt.locator_params(axis='y', nbins=3)

        # plot negative zero and positive ticks of x axis according to best ticks, round down wmin and wmax to 2 decimals
        plt.xticks([np.ceil(wmin * 100) / 100, 0, np.floor(wmax * 100) / 100])
        #format x tickt to 1 decimal
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(key)
        logFormatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False)
        # round kde_array max to the nearest 10th power of 10
        minlogMax = np.floor(np.log10(kde_array.max()))
        minlogMax = 10 ** minlogMax
        # add a colorbar with 3 ticks
        color_ticks = [np.max(kde_array.min(), initial=1e-10), np.max(minlogMax, initial=1e-10)]
        plt.colorbar(ticks=color_ticks, format=logFormatter, label='density')
        # ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) do this for colorbar



plt.tight_layout()
plt.show()

#%%

# import garbage collector
import gc

# collect garbage
gc.collect()
#%%



with mpl.rc_context(rcParams):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

    # plot the histograms for each key in the dictionary
    for ikey, (key, weights) in enumerate(net_weights_dict.items()):
        ax = axes.flatten()[ikey]
        print(weights.shape)
        # get the columns of the weights matrix that correspond to the units
        unit_weights = weights[imagenette_units, :]
        # plot the histograms by row as kde curves
        for irow in range(unit_weights.shape[0]):
            sns.kdeplot(unit_weights[irow, :], ax=ax, label=f'unit {imagenette_units[irow]}', cumulative=False)
        # plot vertical x line at 0
        ax.axvline(0, color='k', linestyle='--')
        # # set ax to log scale
        # ax.set_xscale('symlog')
        # #  ylog
        # ax.set_yscale('symlog')
        ax.set_xlabel('weight')
        ax.set_ylabel('kde')
        ax.set_title(key)
        # one to one aspect ratio in physical units
        ax.set_aspect(1/ax.get_data_ratio())
        # legend outside to the right centered vertically on the whole figure
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0.)
        # log scale
    plt.tight_layout()

#%%




#
# #%%
# # plot the matrix in output tensor 0
# plt.ioff()
# tmp_image = output[0][:, :, 4].detach().cpu().numpy()
# tmp_image = tmp_image.squeeze()
# percent_detected = (tmp_image > 0.25).sum(axis=1) / tmp_image.shape[1]
# # imshow in color log scale
# tmp_array = tmp_image.flatten()
# # threshold to values above 0.25ch
# tmp_array = tmp_array[tmp_array > 0.25]
# # compute cdf of the array
# tmp_array = np.sort(tmp_array)
# cdf = np.cumsum(tmp_array) / tmp_array.sum()
#
# # line plot
# plt.plot(tmp_array, cdf)
# #log log
# # plt.loglog(tmp_array, cdf)
#
#
#
# plt.show()
#
# #%%

#%%