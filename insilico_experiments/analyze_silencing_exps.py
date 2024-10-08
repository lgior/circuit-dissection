import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import torch
import lpips
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid
from torchmetrics.functional import pairwise_euclidean_distance
import re
import pandas as pd
import seaborn as sns

from PIL import Image
from core.utils import make_grid_np, saveallforms, crop_from_montage, summary_by_block
from core.utils.montage_utils import crop_all_from_montage
import itertools
from scipy.stats import bootstrap
# for plotting
from scipy.stats import pearsonr
import seaborn as sns
from scipy.stats import gaussian_kde as kde

import insilico_experiments.analyze_evolutions as anevo


# %% test for the perturbation analysis
torch.cuda.empty_cache()

# Sections to load GAN
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names,
                                       save_as_images)  # install it from huggingface GR
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN

rootdir = r"C:\Users\giordano\Documents\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
rootdir = r"M:\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
rootdir = r"C:\Users\gio\Data"  # since my home folder got full

net_str = 'vgg16' #'alexnet'  # 'resnet50', 'alexnet-eco-080'

if any(s in net_str for s in ['alexnet-eco', 'resnet50']):
    layer_str = '.Linearfc'  # for resnet50 and alexnet-eco-080
else:
    layer_str = '.classifier.Linear6'  # for alexnet

print(f'Possible units to analyze: {anevo.get_recorded_unit_indices(rootdir, net_str, layer_str)}')

unit_idx = 574#398# imagenet 373  # ecoset 13, 373, 14, 12, 72, 66, 78

unit_pattern = net_str + '_' + layer_str + '_' + str(unit_idx)
# unit_pattern = 'alexnet-eco-080.*%s_%s' % (layer_str, unit_idx)
# unit_pattern = 'alexnet_.*%s_%s' % (layer_str, unit_idx)
# unit_pattern = 'resnet50_%s_%s' % (layer_str, unit_idx)
# unit_pattern = 'resnet50_linf8_%s_%s' % (layer_str, unit_idx)
perturbation_pattern = '_kill_topFraction_'
# unit_pattern += perturbation_pattern
figdir = os.path.join('C:', 'Users', 'gio', 'Data', 'figures', 'silencing')
os.makedirs(figdir, exist_ok=True)
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

# %%
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
df = pd.DataFrame.from_dict({key: perturbation_data_dict[key] for key in columns})
df = df.explode('scores')

# %%
df.scores = df.scores.astype("float")
df.type = df.type.astype("string")

# %%
# Get the indices from the flattened lists of lists to retrieve the maximally activating images.
# First reset the index so it runs from 0 to len(flattened_list)
df = df.reset_index(names='old_index')
max_inds = df.sort_values('scores', ascending=False).drop_duplicates(['type', 'strength']).index.values

# %%
# Get list of images from generator
im_generator = (im for images in perturbation_data_dict['images'] for im in images)
max_per_condition_list_gen = []
last_index = -1
for curr_index in np.sort(max_inds):
    max_per_condition_list_gen.append(next(itertools.islice(im_generator, curr_index - last_index - 1, None)))
    last_index = curr_index

# Or simpler with a list comprehension after generating iterable that chains all into a list
# This searches for the index in increasing order so output list will be sorted
max_per_condition_im_list = [im for ind, im in enumerate(itertools.chain.from_iterable(perturbation_data_dict['images']))
                             if ind in max_inds]

#%%
# TODO use this to get top 3 images or top N
# df.sort_values(['job','count'],ascending=False).groupby('job').head(3)

# %%
# both approaches work
all([torch.allclose(gen_l, list_l) for gen_l, list_l in zip(max_per_condition_im_list, max_per_condition_list_gen)])

#%%
# Get all indices from a groupby
# df.groupby(['type', 'strength']).apply(lambda x: list(x.index))
max_per_condition_group = df.groupby(['type', 'strength']).apply(lambda group: group.index[np.argmax(group.scores)])
max_per_condition_df = pd.DataFrame({'max_index': max_per_condition_group}).reset_index()
#%%
# Alternative plotting code. TODO remove it
n_types = len(pd.unique(max_per_condition_df.sort_values(['type', 'strength']).type))
max_n_ims_type = max_per_condition_df.sort_values(['type', 'strength']).groupby('type').apply(len).max()
# for ii, (name, group) in enumerate(max_per_condition_df.sort_values(['type', 'strength']).groupby('type')):
#     # print(group)
#     print(name)
#     for jj, row in enumerate(group.iterrows()):
#         print(row)
#         plt.subplot(n_types, max_n_ims_type, max_n_ims_type * ii + jj + 1)
#         plt.imshow(ToPILImage()(max_per_condition_im_list[df.old_index[row[1].max_index]]))  # index is the 0th column of the row.
# plt.show()
#%%
# Plot the mean image or top image
from mpl_toolkits.axes_grid1 import ImageGrid


max_df = df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength'], group_keys=False).apply(
    lambda group: group.sort_values('scores', ascending=False).head(1))  # set group_keys=False to avoid multiindex creation


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
            if ii == 0 and jj == 0:
                plt.title('(strength: {},\nscore: {:0.2f})'.format(df.strength[merged_list_index], df.scores[merged_list_index]))
            else:
                plt.title('({}, {:0.2f})'.format(df.strength[merged_list_index], df.scores[merged_list_index]))
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
    # plt.show()

# plot max images
fig = plt.figure(figsize=(20., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_types, max_n_ims_type),  # creates 2x2 grid of axes
                 axes_pad=0.4,  # pad between axes in inch.
                 )
plot_images_across_silencing_conditions(max_df, plot_weighted_im_mean=False)
plt.savefig(os.path.join(figdir, '%s__preferred_ims_best.pdf' % original_dir[0]), dpi=300, format='pdf', metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None, transparent=True
           )
plt.show()

# plot average images TODO correct weighted image to proper range e.g. 0-1
fig = plt.figure(figsize=(20., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_types, max_n_ims_type),  # creates 2x2 grid of axes
                 axes_pad=0.4,  # pad between axes in inch.
                 )
plot_images_across_silencing_conditions(max_df, plot_weighted_im_mean=True)
plt.savefig(os.path.join(figdir, '%s__preferred_ims_mean.pdf' % original_dir[0]), dpi=300, format='pdf', metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None, transparent=True
           )
plt.show()
#%%

for i_cond, (scores, images) in enumerate(zip(perturbation_data_dict['scores'], perturbation_data_dict['images'])):
    # print(scores)
    # print(list_argsort(scores, reverse=True)[:9])
    # print([scores[i] for i in list_argsort(scores, reverse=True)[:9]])
    # select top9 activating images, and convert to image grid
    top9_im_grid = anevo.get_top_n_im_grid(scores, images, top_n=9)
# top9_im_grid.show()

#%%
# To get top scoring images, but maybe more efficient to just use the lists and index back as below
# top_9_df = df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength']).apply(lambda group: group.sort_values('scores', ascending=False).head(9))

fig = plt.figure(figsize=(20., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_types, max_n_ims_type),  # creates 2x2 grid of axes
                 axes_pad=0.4,  # pad between axes in inch.
                 )

for ii, (name, group) in enumerate(df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength']).head(1).groupby('type')):
    print(name)
    # print(group.index)
    # print(group.scores)
    print(group.old_index)
    print(group)
    for jj, (_, row_data) in enumerate(group.iterrows()):
        ax = grid[max_n_ims_type * ii + jj]
        print(row_data)
        plt.sca(ax)
        list_index = row_data.old_index
        top9_im_grid = anevo.get_top_n_im_grid(scores=perturbation_data_dict['scores'][list_index],
                                         images=perturbation_data_dict['images'][list_index], top_n=9)
        plt.imshow(top9_im_grid)  # index is the 0th column of the row.
        if ii == 0 and jj == 0:
            plt.title('(strength: {},\nscore: {:0.2f})'.format(row_data.strength, row_data.scores))
        else:
            plt.title('({}, {:0.2f})'.format(row_data.strength, row_data.scores))

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

plt.savefig(os.path.join(figdir, '%s__preferred_ims_top9.pdf' % original_dir[0]), dpi=300, format='pdf', metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None, transparent=True
           )
plt.show()


# %%
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

with mpl.rc_context(params):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    # Plot the strength of response vs strength of input perturbation
    # fig, ax = plt.subplots(figsize=(8,6))
    # df.groupby('type').plot(x='strength', y='scores', kind='scatter', ax=ax)
    ax_line = sns.lineplot(data=df, x='strength', y='scores', hue='type', errorbar=('ci', 95), markers='o',
                           hue_order=['abs', 'inh', 'exc', 'none'],
                           lw=4, err_kws={'edgecolor': None})
    ax_scatter = sns.scatterplot(data=df, x=anevo.jitter(df.strength, 0.008), y='scores', hue='type', alpha=0.5,
                                 hue_order=['abs', 'inh', 'exc', 'none'],
                                 ec='w', s=80)
    # sns.stripplot(df, x='strength', y='scores', hue='type', alpha=0.5) # this makes x axis categorical
    # sns.violinplot(df, x='strength', y='scores', hue='type', inner='points')
    # sns.violinplot(data=df, x="strength", y="scores", inner="points", hue='type')
    # sns.violinplot(data=df[df.type == 'abs'], x="strength", y="scores", inner="points")
    # sns.violinplot(data=df, x='strength', y="scores", hue='type', inner="points", face=0.9)
    h_leg_scatter, label_scatter = ax_scatter.get_legend_handles_labels()
    plt.legend(handles=list(zip(h_leg_scatter[:-4], h_leg_scatter[-4:])), labels=label_scatter[:-4], loc='best',
               labelspacing=0.2,  handlelength=1, fontsize=10)
    # plt.legend()

    plt.title(original_dir[0])
    # ax.set_title('Testing', font=fpath)
    # prop = fm.FontProperties(fname=f.name)
    # ax.set_title('this is a special font:\n%s' % github_url, fontproperties=prop)
    plt.xlabel('Total silencing strength')
    plt.ylabel('Response (a.u.)')
    # plt.ylim([0, 1])
    # ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()

    plt.savefig(os.path.join(figdir, '%s__neuron_response.pdf' % original_dir[0]), dpi=300, format='pdf', metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None, transparent=True
           )
    plt.show()
# %%



# TODO plot the strength of the input with respect to the activations


# %%

# TODO write code to analyze the images
imgs = G.visualize(torch.from_numpy(np.array(fc6_codes)).float().cuda()).cpu()

mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
# mtg_exp.save(join(savedir, "besteachgen%s_%05d.jpg" % (methodlab, RND,)))
mtg = ToPILImage()(make_grid(imgs, nrow=7))
# mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
