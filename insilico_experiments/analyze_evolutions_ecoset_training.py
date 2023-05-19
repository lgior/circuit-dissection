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
# import pandas as pd
from PIL import Image
from core.utils import make_grid_np, saveallforms, crop_from_montage, summary_by_block
from core.utils.montage_utils import crop_all_from_montage
import itertools
from scipy.stats import bootstrap
# for plotting
from scipy.stats import pearsonr
import seaborn as sns
from scipy.stats import gaussian_kde as kde

import insilico_experiments.analyze_evolutions as anevo  # could use ansys


# %%
torch.cuda.empty_cache()

rootdir = r"C:\Users\giordano\Documents\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
rootdir = r"C:\Users\gio\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
layer_str = '.classifier.Linear6'
unit_idx = 373  # 131#373
unit_pattern = 'alexnet-eco-.*%s_%s' % (layer_str, unit_idx)
# %%
data_dirs = [idir for idir in next(os.walk(rootdir))[1] if re.match(unit_pattern, idir)]
# %%
epoch_pattern = 'alexnet-eco-(\d\d\d)_%s_%s' % (layer_str, unit_idx)
epochList = [int(re.search(epoch_pattern, idir).group(1)) for idir in data_dirs if re.match(epoch_pattern, idir)]
# %%
# check if there is a more efficient way to import, and verify all list are in the same order of seeds
for curr_epoch in epochList:
    print('Processing epoch %d from %s' % (curr_epoch, epochList), end='\r')
    epoch_pattern = 'alexnet-eco-%03d_.*' % curr_epoch
    epoch_dir = next((s for s in data_dirs if re.match(epoch_pattern, s)), None)
    anevo.preprocess_artificial_unit_evolution_dir(join(rootdir, epoch_dir), overwrite_existing=True)

# %%
# check if there is a more efficient way to import, and verify all list are in the same order of seeds
dist_mats_list = []
for curr_epoch in epochList:
    print('Processing epoch %d from %s' % (curr_epoch, epochList))
    epoch_pattern = 'alexnet-eco-%03d_.*' % curr_epoch
    epoch_dir = next((s for s in data_dirs if re.match(epoch_pattern, s)), None)
    # Load distance matrices.
    dist_mats_file_path = join(rootdir, epoch_dir, 'distance_mats.npz')
    dist_mats_list.append(np.load(dist_mats_file_path))
# %%
im_dist_mat_list = [dist_mats['arr_0'] for dist_mats in dist_mats_list]
act_dist_mat_list = [dist_mats['arr_1'] for dist_mats in dist_mats_list]
upper_tri_inds = np.triu_indices(im_dist_mat_list[0].shape[0], k=1)
im_dist_triu_vec_list = [mat[upper_tri_inds] for mat in im_dist_mat_list]
act_dist_triu_vec_list = [mat[upper_tri_inds] for mat in act_dist_mat_list]
square_act_dist_triu_vec_list = [act * act for act in act_dist_triu_vec_list]
pearson_corr_obj_list = list(map(pearsonr, im_dist_triu_vec_list, act_dist_triu_vec_list))
pearson_sq_corr_obj_list = list(map(pearsonr, im_dist_triu_vec_list, square_act_dist_triu_vec_list))
pearsonr_list = [pcorr.statistic for pcorr in pearson_corr_obj_list]
pearsonr_sq_list = [pcorr.statistic for pcorr in pearson_sq_corr_obj_list]
pearsonr_ci_list = [list(pcorr.confidence_interval()) for pcorr in pearson_corr_obj_list]
pearsonr_sq_ci_list = [list(pcorr.confidence_interval()) for pcorr in pearson_sq_corr_obj_list]
# %%
fig_dir_path = join(rootdir, 'analysis_figs')
if not os.path.exists(fig_dir_path):
    os.mkdir(fig_dir_path)
# %%
plot_square = True
if plot_square:
    corr_list = pearsonr_sq_list
    corr_ci_list = pearsonr_sq_ci_list
else:
    corr_list = pearsonr_list
    corr_ci_list = pearsonr_ci_list

fig_name = 'pearson'
with plt.rc_context({'font.size': 14}):
    cm = 1 / 2.54
    fig = plt.figure(figsize=(10 * cm, 12 * cm), constrained_layout=True)
    ax = fig.add_subplot(111)
    plot_kws = {'linestyle': '', 'elinewidth': 3, 'markersize': 8, 'marker': 'o'}
    ax.errorbar(epochList, corr_list, yerr=np.abs(np.asarray(corr_ci_list).T - corr_list), **plot_kws)
    plt.xscale('log', base=2)
    ticker_full = mpl.ticker.ScalarFormatter()
    ticker_full.set_scientific(False)
    # ticker_full.format_data('%d')
    ax.xaxis.set_major_formatter(ticker_full)
    plt.xlabel('training epoch')
    plt.ylabel('Pearson\'s r($D_{im}$, $D_{act}$)')
    plt.title('Correlation (95%CI) of image vs activity distances over training', wrap=True)
    plt.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig_file_name = '%s_RSMs_correlations_over_training_alexnet-eco_epochs_1-80_%s_%s.png' % (
    fig_name, layer_str, unit_idx)
    # fig_file_name = 'RSMs_correlations_over_training_alexnet-eco_epochs_1-80_%s_%s.png' % (layer_str.replace('.','-'), unit_idx)
    plt.savefig(join(fig_dir_path, fig_file_name), format='png')
    plt.show()

# %%
pearsonr(epochList, corr_list)
