""" Cluster version of BigGAN Evol """
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
#%%
# if sys.platform == "linux":
#     # rootdir = r"/scratch/binxu/BigGAN_Optim_Tune_new"
#     # Hdir_BigGAN = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
#     # Hdir_fc6 = r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
#     # O2 path interface
#     scratchdir = "/n/scratch3/users/b/biw905"  # os.environ['SCRATCH1']
#     rootdir = join(scratchdir, "GAN_Evol_cmp")
#     Hdir_BigGAN = join("/home/biw905/Hessian", "H_avg_1000cls.npz")  #r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
#     Hdir_fc6 = join("/home/biw905/Hessian", "Evolution_Avg_Hess.npz")  #r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
# else:
#     # rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune_tmp"
#     rootdir = r"D:\Cluster_Backup\GAN_Evol_cmp" #r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
#     Hdir_BigGAN = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
#     Hdir_fc6 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz"

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--layer", type=str, default="fc6", help="Network model to use for Image distance computation")
parser.add_argument("--chans", type=int, nargs='+', default=[0, 25], help="")
parser.add_argument("--G", type=str, default="BigGAN", help="")
parser.add_argument("--optim", type=str, nargs='+', default=["HessCMA", "HessCMA_class", "CholCMA", "CholCMA_prod", "CholCMA_class"], help="")
parser.add_argument("--steps", type=int, default=100, help="")
parser.add_argument("--reps", type=int, default=2, help="")
parser.add_argument("--RFresize", type=bool, default=False, help="")
parser.add_argument("--perturb", type=str, default=None, help="Select perturbation to apply to the network or evolved unit")
# args = parser.parse_args() # ["--G", "BigGAN", "--optim", "HessCMA", "CholCMA","--chans",'1','2','--steps','100',"--reps",'2']
# print(args)
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "fc6", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'6'])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "BigGAN", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'6'])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "fc6", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'10', "--perturb", "kill_topN_in_weight_-1"])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "BigGAN", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'10', "--perturb", "kill_topN_in_weight_1"])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "fc6", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'2', "--perturb", "kill_topFraction_in_weight_0.1"])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "BigGAN", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'2', "--perturb", "kill_topFraction_in_weight_0.1"])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "fc6", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'10', "--perturb", "kill_topFraction_abs_in_weight_0.99"])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.Linear6", "--G", "BigGAN", "--optim", "CholCMA","--chans",'131','132','--steps','100',"--reps",'10', "--perturb", "kill_topFraction_abs_in_weight_0.99"])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.ReLU5", "--G", "fc6", "--optim", "CholCMA","--chans",'1385','1386','--steps','100',"--reps",'10'])
# args = parser.parse_args(["--net", "alexnet-eco-080", "--layer", ".classifier.ReLU5", "--G", "BigGAN", "--optim", "CholCMA","--chans",'1385','1386','--steps','100',"--reps",'10'])
# args = parser.parse_args(["--net", "resnet50", "--layer", ".Linearfc", "--G", "fc6", "--optim", "CholCMA","--chans",'373','374','--steps','100',"--reps",'10'])
#%%
"""with a correct cmaes or initialization, BigGAN can match FC6 activation."""
# Folder to save
if os.environ['COMPUTERNAME'] == 'MNB-PONC-D21184':  # new pc
    rootdir = r"M:\Data"
    rootdir = r"C:\Users\gio\Data"  # personal folder gets full at 50GB
else:
    rootdir = r"C:\Users\giordano\Documents\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"

#%% Select GAN


def load_GAN(name):
    if name == "BigGAN":
        BGAN = BigGAN.from_pretrained("biggan-deep-256")
        BGAN.eval().cuda()
        for param in BGAN.parameters():
            param.requires_grad_(False)
        G = BigGAN_wrapper(BGAN)
    elif name == "fc6":
        G = upconvGAN("fc6")
        G.eval().cuda()
        for param in G.parameters():
            param.requires_grad_(False)
    else:
        raise ValueError("Unknown GAN model")
    return G


def load_Hessian(name):
    # Select Hessian
    try:
        if name == "BigGAN":
            H = np.load(Hdir_BigGAN)
        elif name == "fc6":
            H = np.load(Hdir_fc6)
        else:
            raise ValueError("Unknown GAN model")
    except:
        print("Hessian not found for the specified GAN")
        H = None
    return H


#%% Optimizer from label, Use this to translate string labels to optimizer

def resize_and_pad(imgs, corner, size):
    """ Resize and pad images to a square with a given corner and size
    Background is gray.
    Assume image is float, range [0, 1]
    """ # FIXME: this should depend on the input size of image, add canvas size parameter
    pad_img = torch.ones_like(imgs) * 0.5
    rsz_img = F.interpolate(imgs, size=size, align_corners=True, mode="bilinear")
    pad_img[:, :, corner[0]:corner[0]+size[0], corner[1]:corner[1]+size[1]] = rsz_img
    return pad_img
#%%

# TODO refactor following repeated code into another function or method
rootdir = r"C:\Users\gio\Data"  # since my home folder got full
# layer_str = '.classifier.Linear6'
layer_str = '.Linearfc' # for resnets
unit_idx = 574#398# imagenet 373  # ecoset 13, 373, 14, 12, 72, 66, 78
unit_pattern = 'alexnet-eco-080.*%s_%s' % (layer_str, unit_idx)
# unit_pattern = 'alexnet_.*%s_%s' % (layer_str, unit_idx)
unit_pattern = 'resnet50_%s_%s' % (layer_str, unit_idx)
unit_pattern = 'resnet50_linf0.5_%s_%s' % (layer_str, unit_idx)
unit_pattern = 'resnet50_linf8_%s_%s' % (layer_str, unit_idx)
perturbation_pattern = '_kill_topFraction_'
# %%
original_dir = [idir for idir in next(os.walk(rootdir))[1] if re.match(unit_pattern + '$', idir)]
data_dirs = [idir for idir in next(os.walk(rootdir))[1] if re.match(unit_pattern + perturbation_pattern, idir)]
# %%
perturbation_fractions = [float(re.search(r'.*_([\-\.\d]+)?', dir).group(1)) for dir in data_dirs]
perturbation_types = ['abs' if 'abs' in data_dir else 'exc' if fraction >= 0 else 'inh'
                      for data_dir, fraction in zip(data_dirs, perturbation_fractions)]
# add original unperturbed unit
perturbation_fractions.append(0)
perturbation_types.append('none')
perturbation_data_dict = {'type': perturbation_types,
                          'strength': [abs(frac) for frac in perturbation_fractions]}
#%%
# Extract the top image and score per evolution experiment inside a folder, dimension n_folders x m_evolutions
max_scores_list = []
max_im_tensor_list = []
for data_dir in data_dirs + original_dir:
    artificial_unit_dir = join(rootdir, data_dir)
    max_scores, max_im_tensor = anevo.extrac_top_scores_and_images_from_dir(artificial_unit_dir)
    print("%s has %d scores" % (data_dir, len(max_scores)))
    max_scores_list.append(max_scores)
    max_im_tensor_list.append(max_im_tensor)


# %%
# plot the amount of silencing versus the maximum scores
perturbation_data_dict['scores'] = max_scores_list
perturbation_data_dict['images'] = max_im_tensor_list

# %%

columns = ['type', 'strength', 'scores']
# columns = ['type', 'strength', 'scores', 'images']
df = pd.DataFrame.from_dict({key: perturbation_data_dict[key] for key in columns})
df = df.explode(['scores'])
# df = df.explode(['scores', 'images'])

#%%
df.scores = df.scores.astype("float")
df.type = df.type.astype("string")

# %%
# Get the indices from the flattened lists of lists to retrieve the maximally activating images.
# First reset the index so it runs from 0 to len(flattened_list)
df = df.reset_index(names='old_index')
max_inds = df.sort_values('scores', ascending=False).drop_duplicates(['type', 'strength']).index.values

#%%
# Get maximum image per condition
max_df = df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength'], group_keys=False).apply(
    lambda group: group.sort_values('scores', ascending=False).head(1))  # set group_keys=False to avoid multiindex creation

#%%
max_per_condition_im_list = [im for ind, im in enumerate(itertools.chain.from_iterable(perturbation_data_dict['images']))
                             if ind in max_inds]
# Sort so the order matches the max_df
max_per_condition_im_list = [max_per_condition_im_list[sorted(max_inds).index(ind)] for ind in max_df.index]
#%%
topn = 10
topn_df = df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength']).apply(
    lambda group: group.sort_values('scores', ascending=False).head(topn))
#%%
# Get the list of top 5 images sorted in same order as dataframe

# Select only one index of multiindex DataFrame
# https://stackoverflow.com/questions/28140771/select-only-one-index-of-multiindex-dataframe
top_indices = topn_df.index.get_level_values(2).values

top_per_condition_im_list = [im
                             for ind, im in enumerate(itertools.chain.from_iterable(perturbation_data_dict['images']))
                             if ind in top_indices]
top_per_condition_im_list = [top_per_condition_im_list[sorted(top_indices).index(ind)] for ind in top_indices]
#%%
# process group by group if number of images becomes too large

# for group_key, new_df in topn_df.groupby(level=[0, 1]):
#      print(group_key)
#      print(new_df)
#      print(new_df.index.get_level_values(2).values)
#      break

# %%
# Get image tensor and mean output activations per experiment condition
only_max_im = False
if only_max_im:
    # Conver the images from pandas to numpy by df.values, this returns an array of dtype object, use numpy to concatenate to a float array,
    # then finally convert it to torch tensor
    # all_ims_tensor = torch.from_numpy(np.stack(max_df['images'].values))  # not a good idea to put tensors into pandas
    all_ims_tensor = torch.stack(max_per_condition_im_list)
else:
    all_ims_tensor = torch.stack(top_per_condition_im_list)

network_list = ['alexnet',
                'resnet50', 'resnet50_linf0.5', 'resnet50_linf1', 'resnet50_linf2', 'resnet50_linf4', 'resnet50_linf8']
model_outputs = []
# Now pass the tensor to get network activations

for net in network_list:
    # scorer = TorchScorer(args.net)
    scorer = TorchScorer(net)
    model_output = scorer.model(all_ims_tensor.to(torch.device('cuda:0'))).cpu().detach()
    model_output = torch.nn.functional.softmax(model_output, dim=1)
    # we know we picked topn images per group
    if topn > 1:
        model_output = model_output.reshape(-1, topn, 1000).mean(axis=1)
    model_outputs.append(torch.log(model_output))  # cross entropy loss uses sum over one-hot class x log(softmax(class))

#%%

control_index = np.where(max_df.type == 'none')[0][0]
    # compute the whole correlation matrix but keep only the line with respect to the control image
correlations_df = pd.DataFrame({net: np.corrcoef(model_outputs[_i])[control_index, :] for _i, net in enumerate(network_list)})
correlations_df = correlations_df.assign(strength=max_df.strength.values)
correlations_df = correlations_df.assign(type=max_df.type.values)
#%%
# sns.heatmap(correlations_df[correlations_df.type=='inh'].drop(columns=['type']))
# plt.show()
# sns.lineplot(data=correlations_df[correlations_df.type=='inh'].drop(columns=['type']),
#              y=correlations_df.drop(columns=['strength','type']).columns.values, x='strength')
corr_melt_df = correlations_df.melt(['strength', 'type'], var_name='net', value_name='correlations')
sns.lineplot(data=corr_melt_df[corr_melt_df.type=='inh'], y='correlations', x='strength', hue='net', palette='Dark2_r')
plt.show()
sns.lineplot(data=corr_melt_df[corr_melt_df.type=='exc'], y='correlations', x='strength', hue='net', palette='Dark2_r')
plt.show()
sns.lineplot(data=corr_melt_df[corr_melt_df.type=='abs'], y='correlations', x='strength', hue='net', palette='Dark2_r')
plt.show()
# max_df['corr_w_ctrl'] = np.corrcoef(model_output)[control_index, :]
#%%
from tempfile import NamedTemporaryFile
import urllib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path
github_url = 'https://github.com/google/fonts/blob/master/ofl/alike/Alike-Regular.ttf'

url = github_url + '?raw=true'  # You want the actual file, not some html

response = urllib.request.urlopen(url)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()

fpath = Path(mpl.get_data_path(), f.name)
#%%
params = {
   'axes.labelsize': 16,
   'font.size': 28,
   'font.family': 'Arial',
   'legend.fontsize': 16,
   'xtick.labelsize': 22,
   'ytick.labelsize': 22,
   'text.usetex': False,
   'figure.figsize': [5, 5]
   }
# rcParams.update(params)
figdir = os.path.join('C:', 'Users', 'gio', 'Data', 'figures', 'silencing')
os.makedirs(figdir, exist_ok=True)
with mpl.rc_context(params):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    # fig = plt.figure(figsize=(2, 2), dpi=300)
    # show single lines
    # sns.lineplot(data=corr_melt_df, y='correlations', x='strength', hue='type', hue_order=['abs', 'inh', 'exc'],
    #              units='net', estimator=None, lw=0.5)
    sns.lineplot(data=corr_melt_df, y='correlations', x='strength', hue='type', hue_order=['abs', 'inh', 'exc'],
                 lw=3, marker='o', mec=None, ms=12, err_kws={'edgecolor': None}, errorbar=('ci', 95))

    plt.title(original_dir[0])
    # ax.set_title('Testing', font=fpath)
    # prop = fm.FontProperties(fname=f.name)
    # ax.set_title('this is a special font:\n%s' % github_url, fontproperties=prop)
    plt.xlabel('Total silencing strength')
    plt.ylabel('Response correlation \n control vs silencing')
    plt.ylim([0, 1])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()

    #show plot per condition
    # sns.relplot(
    #     data=corr_melt_df, y='correlations', x='strength', col='type',
    #     hue='type', hue_order=['abs', 'inh', 'exc'], col_order=['inh', 'exc', 'abs'], #  style='net'
    #     kind="line"
    # )
    plt.savefig(os.path.join(figdir, '%s__net_output_correlations.pdf' % original_dir[0]), dpi=300, format='pdf', metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None, transparent=True
           )
    plt.show()
# %%
# sns.barplot(correlations_df.drop(columns=['type', 'strength']))
bar_h = sns.barplot(data=corr_melt_df, y='correlations', x='net', hue='type', hue_order=['abs', 'inh', 'exc'])
bar_h.set_xticklabels(bar_h.get_xticklabels(), rotation=15, horizontalalignment='right')
plt.show()


d3dds
#%%
norm = plt.Normalize(max_df.strength[max_df.type=='inh'].min(), max_df.strength[max_df.type=='inh'].max())
smap = plt.cm.ScalarMappable(cmap='viridis', norm=norm)

cmap = mpl.cm.ScalarMappable().get_cmap()
for _ind in zip(*np.where((max_df.type == 'inh').values.astype(bool))):
    print(_ind)
    plt.scatter(model_output[control_index, :], model_output[_ind, :], c=cmap(max_df.strength.iloc[_ind]))

plt.show()

#%%

norm = plt.Normalize(max_df.strength[max_df.type == 'exc'].min(), max_df.strength[max_df.type == 'exc'].max())
smap = plt.cm.ScalarMappable(cmap='viridis', norm=norm)

cmap = mpl.cm.ScalarMappable().get_cmap()
for _ind in zip(*np.where((max_df.type == 'exc').values.astype(bool))):
    print(_ind)
    plt.scatter(model_output[control_index, :], model_output[_ind, :], color=cmap(max_df.strength.iloc[_ind]))

plt.show()

#%%
import colorsys


def lighten_color_hsl(color, amount):
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(color))
    rgb_c = colorsys.hls_to_rgb(c[0], max(0, min(1, amount + c[1])), c[2])
    return [min(1, max(0, x)) for x in rgb_c]


perturb_type = 'abs'
baseCmap = plt.get_cmap("Greens_r")


def plot_activations_scatter_silencing_vs_control(perturb_type, baseCmap):
    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(max_df.strength[max_df.type == perturb_type].min(),
                                max_df.strength[max_df.type == perturb_type].max())
    # smap = plt.cm.ScalarMappable(cmap=mpl.colormaps.get_cmap('Greens'), norm=norm)
    # cmap = mpl.cm.ScalarMappable().get_cmap()
    lightness_factor = -0.11
    # to get some listed map of colors matching the discrete values but with continuous norm
    strengths = max_df.strength[max_df.type == perturb_type].values
    cmaplist = [lighten_color_hsl(baseCmap(norm(x)), lightness_factor) for x in strengths]
    cmap = mpl.colors.ListedColormap(cmaplist, len(strengths))
    levelNorm = mpl.colors.BoundaryNorm(np.hstack([strengths, 1.05]) - 0.05, cmap.N)

    plt.axline((0, 0), slope=1, color='gray')
    for _ind in zip(*np.where((max_df.type == perturb_type).values.astype(bool))):

        icolor = lighten_color_hsl(baseCmap(norm(max_df.strength.iloc[_ind])), lightness_factor)
        print('%s original %s' % (icolor, baseCmap(norm(max_df.strength.iloc[_ind]))))
        # TODO change 0 index to index of df from string search.
        h_map = ax.scatter(model_output[control_index, :], model_output[_ind, :], color=icolor, s=14)
    fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=levelNorm), ticks=strengths)
    plt.show()

plot_activations_scatter_silencing_vs_control('inh', plt.get_cmap('Oranges_r'))
plot_activations_scatter_silencing_vs_control('exc', plt.get_cmap('Greens_r'))
plot_activations_scatter_silencing_vs_control('abs', plt.get_cmap('Blues_r'))

#%%

# TODO do this for the average of all the observer networks
import seaborn as sns


# compute similarities of the activations
corr_mat = np.corrcoef(model_output[control_index, :], model_output[np.where((max_df.type == 'inh').values.astype(bool))])
sns.heatmap(corr_mat, cmap='PuOr_r', vmin=-1, vmax=1)
plt.show()

corr_mat = np.corrcoef(model_output[control_index, :], model_output[np.where((max_df.type == 'exc').values.astype(bool))])
sns.heatmap(corr_mat, cmap='PRGn', vmin=-1, vmax=1)
plt.show()

corr_mat = np.corrcoef(model_output[control_index, :], model_output[np.where((max_df.type == 'abs').values.astype(bool))])
sns.heatmap(corr_mat, cmap='RdBu', vmin=-1, vmax=1)
plt.show()
#
# corr_mat = np.corrcoef(model_output[np.hstack(([control_index], np.squeeze(np.where((max_df.type == 'exc').values.astype(bool))))), :])
# sns.heatmap(corr_mat)
# plt.show()
#%%
# corr_dict = {k: np.corrcoef(model_output[0, :], model_output[np.where((max_df.type == k).values.astype(bool))])[0, 1:]
#              for k in ['abs', 'exc', 'inh']}


#%%
sns.heatmap(np.corrcoef(model_output))
plt.show()
#%%
# sns.lineplot(data=)

ax_scatter = sns.lineplot(data=max_df, x='strength', y='corr_w_ctrl', hue='type', alpha=0.5)
plt.show()

#%%
# Check the softmax of the output
softmax = torch.exp(model_outputs[0])
sns.heatmap(softmax, cmap='viridis')
plt.show()

# this is log(softmax(x)) log(p)
sns.heatmap(model_outputs[0], cmap='plasma_r')
plt.show()
#%%
# change two columns of pandas into single string https://stackoverflow.com/questions/11858472/string-concatenation-of-two-pandas-columns
sns.heatmap(model_output.cpu().detach(),
            yticklabels=(max_df.type + '_' + max_df.strength.astype(str)))
plt.show()
#%%
# GR insert perturbation to network
if args.perturb is None:
    weights_silenced = None
elif re.search(r'kill_topN_in_weight_(.*)', args.perturb):
    n_in_to_kill = int(re.search(r'kill_topN_in_weight_(.*)', args.perturb).group(1))
    target_module, original_weights, weights_silenced = \
        kill_fc_top_in_n(scorer.model, args.layer, unit_id, n_in_to_kill)
elif re.search(r'kill_topFraction_in_weight_(.*)', args.perturb):
    silencing_fraction = float(re.search(r'kill_topFraction_in_weight_(.*)', args.perturb).group(1))
    target_module, original_weights, weights_silenced = \
        kill_fc_top_in_fraction(scorer.model, args.layer, unit_id, silencing_fraction)
elif re.search(r'kill_topFraction_abs_in_weight_(.*)', args.perturb):
    silencing_fraction = float(re.search(r'kill_topFraction_abs_in_weight_(.*)', args.perturb).group(1))
    target_module, original_weights, weights_silenced = \
        kill_fc_top_in_fraction(scorer.model, args.layer, unit_id, silencing_fraction, do_abs=True)
else:
    raise NotImplementedError

#%%

#%%
def plot_images_across_silencing_conditions(max_df, plot_weighted_im_mean=False):
    for ii, (name, group) in enumerate(max_df.groupby('type')):
        print(group)
        for jj, row in enumerate(group.iterrows()):
            # if jj % 2 == 0:  # print every second image
            # print(row[1])
            # print(row[0])
            ax = grid[max_n_ims_type * ii + jj]
            plt.sca(ax)
            merged_list_index, df_columns = row
            list_index = df_columns.old_index
            if plot_weighted_im_mean:
                weights = torch.tensor(perturbation_data_dict['scores'][list_index]).float()
                im_tensor = perturbation_data_dict['images'][list_index]
                weighted_mean_image = ToPILImage()((weights.reshape((-1, 1, 1, 1)) * im_tensor).sum(axis=0) / weights.sum())
                image = weighted_mean_image
            else:
                image = ToPILImage()(max_per_condition_im_list[list_index])  # index is the 0th column of the row.
                # image = ToPILImage()(perturbation_data_dict['images'][list_index])  # index is the 0th column of the row.
            plt.imshow(image)  # index is the 0th column of the row.
            plt.title('strength: {},\n score:{:0.2f}'.format(df.strength[merged_list_index], df.scores[merged_list_index]))
            if jj == 0:
                ax.set_ylabel(name)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                plt.axis('off')
        # else:
        #     ax = grid[max_n_ims_type * ii + jj]
        #     ax.remove()
    for ax in grid[max_n_ims_type * ii + jj + 1:]:
        plt.sca(ax)
        plt.axis('off')
        # maybe remove instead of setting it off
    plt.show()

#%%
import analyze_evolutions as anevo
#TODO
# 1) Import the networks
# 2) read the images from stored evolutions
# 3) pass the images to the different networks
# Store the preprocesed data in a new file to save processing time

# data_dict_list = anevo.load_folder_evolutions(join(rootdir, original_dir[0])) # loads all evolutions inside a folder

#%%
G = load_GAN(args.G)
Hdata = load_Hessian(args.G)
#%% Select vision model as scorer
scorer = TorchScorer(args.net)
# net = tv.alexnet(pretrained=True)
# scorer.select_unit(("alexnet", "fc6", 2))
# imgs = G.visualize(torch.randn(3, 256).cuda()).cpu()
# scores = scorer.score_tsr(imgs)
#%% Select the Optimizer
method_col = args.optim
# optimizer_col = [label2optimizer(methodlabel, np.random.randn(1, 256), GAN=args.G) for methodlabel in method_col]
#%% Set recording location and image size and position.
pos_dict = {"conv5": (7, 7), "conv4": (7, 7), "conv3": (7, 7), "conv2": (14, 14), "conv1": (28, 28)}

# Get the center position of the feature map.
if not "fc" in args.layer and not ".classifier" in args.layer:
    # if not args.net in layername_dict:  # TODO:Check the logic
        module_names, module_types, module_spec = get_module_names(scorer.model, input_size=(3, 227, 227), device="cuda")
        layer_key = [k for k, v in module_names.items() if v == args.layer][0]
        feat_outshape = module_spec[layer_key]['outshape']
        assert len(feat_outshape) == 3  # fc layer will fail
        cent_pos = (feat_outshape[1]//2, feat_outshape[2]//2)
    # else:
    #     cent_pos = pos_dict[args.layer]
else:
    cent_pos = None

print("Target setting network %s layer %s, center pos"%(args.net, args.layer), cent_pos)
# rf Mapping,
if args.RFresize and not "fc" in args.layer:
    print("Computing RF by direct backprop: ")
    gradAmpmap = grad_RF_estimate(scorer.model, args.layer, (slice(None), *cent_pos), input_size=(3, 227, 227),
                                  device="cuda", show=False, reps=30, batch=1)
    Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
    corner = (Xlim[0], Ylim[0])
    imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
else:
    imgsize = (256, 256)
    corner = (0, 0)
    Xlim = (corner[0], corner[0] + imgsize[0])
    Ylim = (corner[1], corner[1] + imgsize[1])

print("Xlim %s Ylim %s \n imgsize %s corner %s" % (Xlim, Ylim, imgsize, corner))

#%% Start iterating through channels.
for unit_id in range(args.chans[0], args.chans[1]):
    if "fc" in args.layer or "classifier" in args.layer:
        unit = (args.net, args.layer, unit_id)
    else:
        unit = (args.net, args.layer, unit_id, *cent_pos)
    scorer.select_unit(unit)


    # Save directory named after the unit. Add RFrsz as suffix if resized
    if cent_pos is None:
        savedir = join(rootdir, r"%s_%s_%d" % unit[:3])
    else:
        savedir = join(rootdir, r"%s_%s_%d_%d_%d" % unit[:5])

    if args.RFresize: savedir += "_RFrsz"
    # GR add suffix if perturbed
    if args.perturb is not None:
        if len(weights_silenced) > 5:
            savedir += '_' + args.perturb
        else:
            savedir += '_' + args.perturb + '_' + '-'.join(map(str, weights_silenced))
    os.makedirs(savedir, exist_ok=True)
    for triali in range(args.reps):
        # generate initial code.
        if args.G == "BigGAN":
            fixnoise = 0.7 * truncated_noise_sample(1, 128)
            init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
        elif args.G == "fc6":
            init_code = np.random.randn(1, 4096)
        RND = np.random.randint(1E5)
        np.save(join(savedir, "init_code_%05d.npy"%RND), init_code)
        optimizer_col = [label2optimizer(methodlabel, init_code, args.G) for methodlabel in method_col]
        for methodlab, optimizer in zip(method_col, optimizer_col):
            if args.G == "fc6":  methodlab += "_fc6"  # add space notation as suffix to optimizer
            # core evolution code
            new_codes = init_code
            # new_codes = init_code + np.random.randn(25, 256) * 0.06
            scores_all = []
            generations = []
            codes_all = []
            best_imgs = []
            for i in range(args.steps,):
                codes_all.append(new_codes.copy())
                latent_code = torch.from_numpy(np.array(new_codes)).float()
                # imgs = G.visualize_batch_np(new_codes) # B=1
                imgs = G.visualize(latent_code.cuda()).cpu()
                if args.RFresize: imgs = resize_and_pad(imgs, corner, imgsize) # Bug: imgs are resized to 256x256 and it will be further resized in score_tsr
                scores = scorer.score_tsr(imgs)
                if args.G == "BigGAN":
                    print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                        i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                        latent_code[:, :128].norm(dim=1).mean()))
                else:
                    print("step %d score %.3f (%.3f) (norm %.2f )" % (
                        i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
                new_codes = optimizer.step_simple(scores, new_codes, )
                scores_all.extend(list(scores))
                generations.extend([i] * len(scores))
                best_imgs.append(imgs[scores.argmax(),:,:,:])

            codes_all = np.concatenate(tuple(codes_all), axis=0)
            scores_all = np.array(scores_all)
            generations = np.array(generations)
            mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
            mtg_exp.save(join(savedir, "besteachgen%s_%05d.jpg" % (methodlab, RND,)))
            mtg = ToPILImage()(make_grid(imgs, nrow=7))
            mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
            # save_imgrid(imgs, join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())), nrow=7)
            # save_imgrid(best_imgs, join(savedir, "bestgen%s_%05d.jpg" % (methodlab, RND, )), nrow=10)
            # TODO move the saving of silenced weights to outside the loop as is a constant across trials
            if args.G == "fc6":
                np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
                 generations=generations, scores_all=scores_all, codes_fin=codes_all[-80:, :], weights_silenced=weights_silenced)
            else:
                np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
                 generations=generations, scores_all=scores_all, codes_all=codes_all, weights_silenced=weights_silenced)
    # GR recover original weights, so that perturbations do not accumulate with each unit.
    if args.perturb is not None:
        with torch.no_grad():
            target_module.weight = torch.nn.Parameter(original_weights.to('cuda'))

