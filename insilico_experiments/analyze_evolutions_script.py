import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid
import re
# import pandas as pd
from PIL import Image
from core.utils import make_grid_np, saveallforms, crop_from_montage, summary_by_block
from core.utils.montage_utils import crop_all_from_montage
import itertools
#%%
rootdir = r"C:\Users\giordano\Documents\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"

layer_str = '.classifier.Linear6'
unit_idx = 131#373
#%%
data_dirs = [idir for idir in next(os.walk(rootdir))[1] if re.match('alexnet-eco-.*%s_%s'%(layer_str, unit_idx), idir)]

#%%
epochList = [1, 80]
#%%
# check if there is a more efficient way to import, and verify all list are in the same order of seeds
curr_epoch = 80
epoch_dir = next((s for s in data_dirs if re.match('alexnet-eco-%03d_.*' % curr_epoch, s)), None)
epoch_npzfiles = [s for s in os.listdir(join(rootdir, epoch_dir)) if re.match('scores.*', s)]
epoch_seeds = [re.search('.*\_(\d*)\.jpg', s).group(1) for s in os.listdir(join(rootdir, epoch_dir)) if re.match('.*\_\d*\.jpg', s)]
epoch_final_imgfiles = [s for s in os.listdir(join(rootdir, epoch_dir)) if re.match('last.*\_\d*.*\.jpg', s)]
epoch_traj_imgfiles = [s for s in os.listdir(join(rootdir, epoch_dir)) if re.match('best.*\_\d*\.jpg', s)]
initcode_files = [s for s in os.listdir(join(rootdir, epoch_dir)) if re.match('init.*\_\d*\.npy', s)]

scores_list = []
generations_list = []
final_img_list = []
traj_img_list = []
init_codes_list = []
for _ifile in range(len(epoch_npzfiles)):
    evo_data = np.load(join(rootdir, epoch_dir, epoch_npzfiles[_ifile]))
    scores_list.append(evo_data['scores_all'])
    generations_list.append(evo_data['generations'])
    # TODO automatically load correct number of files, check the length of the other vectors.
    if "fc6" in epoch_final_imgfiles[_ifile]:
        num_ims = 40
    else:
        num_ims = 28
    final_img_list.append(
        crop_all_from_montage(
            np.asarray(Image.open(join(rootdir, epoch_dir, epoch_final_imgfiles[_ifile]))), imgsize=256, totalnum=num_ims))
    traj_img_list.append(
        crop_all_from_montage(
            np.asarray(Image.open(join(rootdir, epoch_dir, epoch_traj_imgfiles[_ifile]))), imgsize=256, totalnum=num_ims))
    init_codes_list.append(np.load(join(rootdir, epoch_dir, initcode_files[_ifile])))
    # it seems the image grids have all padding of 2 pixels by default.

#%%
from collections import Counter
Counter(generations_list[0]).values()

#%%

from scipy.stats import bootstrap


def get_mean_ci(data_vec):
    return bootstrap((data_vec,), np.mean, confidence_level=0.95, n_resamples=10).confidence_interval

# def get_exp_summary_dicts(generations_list, scores_list):
exp_dict_list = []
for igen, curr_gens in enumerate(generations_list):
    unique_steps = np.unique(curr_gens)
    score_vec = np.array(scores_list[igen])
    mean_scores = np.array([score_vec[curr_gens == step].mean() for step in unique_steps])
    max_scores = np.array([score_vec[curr_gens == step].max() for step in unique_steps])
    min_scores = np.array([score_vec[curr_gens == step].min() for step in unique_steps])
    std_scores = np.array([score_vec[curr_gens == step].std() for step in unique_steps])
    lo_ci, up_ci = map(list, zip(*[get_mean_ci(score_vec[curr_gens == step]) if step != 0
                                   else (scores_list[igen][curr_gens == step][0],)*2
                                   for step in unique_steps]))
    exp_dict_list.append(dict(
        scores=dict(max=max_scores, min=min_scores, mean=mean_scores, std=std_scores, lo_ci=lo_ci, up_ci=up_ci),
        generations=unique_steps,
        generator=re.search(r'besteachgen(.*)\.jpg', epoch_traj_imgfiles[igen]).group(1)))
    # return exp_dict_list



final_scores_list = []
for igen, curr_gens in enumerate(generations_list):
    score_vec = np.array(scores_list[igen])
    final_scores = score_vec[curr_gens == curr_gens[-1]]
    final_scores_list.append(final_scores)
    final_scores_norm = (final_scores - np.min(final_scores)) / (np.max(final_scores) - np.min(final_scores))
    # weigh images by scores using the alpha channel
    alpha_final_im_list = []
    for f_im, f_sc in zip(final_img_list[igen], final_scores_norm):
        _tmp_im = Image.fromarray(f_im)
        _tmp_im.putalpha(np.int32(np.floor(255 * f_sc + 10)))
        alpha_final_im_list.append(ToTensor()(_tmp_im))
    # sort images by scores
    alpha_final_im_list = [im for _, im in sorted(zip(final_scores, alpha_final_im_list), reverse=True, key=lambda pair: pair[0])]
    im_alpha_grid = ToPILImage()(make_grid(alpha_final_im_list, nrow=7))


def plot_activity_trajectories_ci(list_of_score_dicts):
    ax = plt.axes()
    n_plots = len(list_of_score_dicts)
    cmap = mpl.colormaps['Dark2'].resampled(n_plots)
    for iplot, data in enumerate(list_of_score_dicts):
        ax.fill_between(data['generations'],
                        data['scores']['lo_ci'],
                        data['scores']['up_ci'], color=cmap(iplot), alpha=0.5, label=data['generator'])
    # up_err = max_scores - mean_scores
    # lo_err = np.zeros_like(max_scores)
    # plt.scatter(generations_list[igen], scores_list[igen], color=cmap(igen), alpha=0.05)
    # plt.plot(unique_steps, mean_scores, color=cmap(igen))
    # plt.plot(unique_steps, max_scores, color=cmap(igen))
    # plt.fill_between(unique_steps, mean_scores - std_scores, mean_scores + std_scores, color=cmap(igen), alpha=0.5)
    # ax2.imshow(final_img_list[igen][0])
    # plt.errorbar(unique_steps, mean_scores, yerr=(lo_err, up_err), color=cmap(igen))
    plt.title('95% CI of the mean activity per generation')
    plt.xlabel('generation')
    plt.ylabel('activity')
    legend_h = plt.legend(loc='upper left', fontsize='small', frameon=False, handlelength=1)
    for hl in legend_h.get_lines():
        hl._legmarker.set_markersize(1)
    plt.show()


# plot_activity_trajectories_ci(exp_dict_list)
#%%

all_final_ims = list(itertools.chain.from_iterable(final_img_list))
all_final_scores = list(itertools.chain.from_iterable(final_scores_list))
all_final_scores_norm = (all_final_scores - np.min(all_final_scores)) / (np.max(all_final_scores) - np.min(all_final_scores))
all_alpha_final_ims = []
# weigh by setting transparency according to elicited response
for f_im, f_sc in zip(all_final_ims, all_final_scores_norm):
    _tmp_im = Image.fromarray(f_im)
    _tmp_im.putalpha(np.int32(np.floor(255 * f_sc + 10)))
    all_alpha_final_ims.append(ToTensor()(_tmp_im))

all_alpha_final_ims = [im for _, im in sorted(zip(all_final_scores, all_alpha_final_ims),
                                              reverse=True, key=lambda pair: pair[0])]
all_im_alpha_grid = ToPILImage()(make_grid(all_alpha_final_ims,
                                           nrow=np.int32(np.floor(np.sqrt(len(all_final_scores))))))
# all_im_alpha_grid.show()

#%%
import lpips
from torchvision import transforms

cuda0 = torch.device('cuda:0')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

loss_fn_alex = lpips.LPIPS(net='alex')

#%%
ims_tensor = torch.stack(list(map(ToTensor(), all_final_ims)))
norm_ims_tensor = 2 * ims_tensor - 1
#%%
# We want to generate a distance matrix, so let's loop over diagonals to use the batch processing or pytorch
# alternatively use a list of pairs of indices and process it in batches.
lpips_dist_mat = np.zeros((len(all_final_ims),)*2)

for idiag in range(len(all_final_ims)):
    print(idiag)
    row_inds = np.arange(len(all_final_ims) - idiag)
    # inds_list = (list(zip(row_inds, row_inds + idiag)))
    lpips_dist_mat[row_inds, row_inds + idiag] = loss_fn_alex.forward(norm_ims_tensor[row_inds, :, :, :],
                                                                      norm_ims_tensor[row_inds + idiag, :, :, :]).squeeze().detach().numpy()
    if idiag == 2: break
# print((range(im, len(all_final_ims)), range(im + 1, len(all_final_ims)))

#%%
# This chunk is maybe very inefficient way to write this.

# def normalize_im_pm_one(im_numpy):
#     norm_im = transforms.Normalize(0.5, 0.5)(ToTensor()(im_numpy))
#     return norm_im


# lpips_dist_mat = np.zeros((len(all_final_ims),)*2)
# for i_im in range(4+0*len(all_final_ims)):
#     for j_im in range(i_im + 1, len(all_final_ims)):
#         lpips_dist_mat[i_im, j_im] = loss_fn_alex.forward(normalize_im_pm_one(all_final_ims[i_im]),
#                                                           normalize_im_pm_one(all_final_ims[j_im]))
#%%
lpips_dist_mat_full = lpips_dist_mat + lpips_dist_mat.T

#%%
plt.imshow(lpips_dist_mat_full)
plt.colorbar()
plt.show()
#%%
# compute activity distance, just the diff
from torchmetrics.functional import pairwise_euclidean_distance


pw_euclid_dist_mat = pairwise_euclidean_distance(torch.asarray(all_final_scores_norm).unsqueeze(-1))
plt.imshow(pw_euclid_dist_mat)
plt.colorbar()
plt.show()
#%%
from scipy.stats import pearsonr
import seaborn as sns
# import statsmodels
# Trick to generate colors from a density approximation
from scipy.stats import gaussian_kde as kde

upper_tri_inds = np.triu_indices(pw_euclid_dist_mat.shape[0], k=1)
act_dist_vec = pw_euclid_dist_mat[upper_tri_inds].numpy()
im_dist_vec = lpips_dist_mat_full[upper_tri_inds]
kdeObj = kde(np.vstack((im_dist_vec, act_dist_vec)))


def get_colors(vals):
    colors = np.zeros((len(vals),3))
    norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())
    colors = [mpl.cm.ScalarMappable(norm=norm, cmap='magma_r').to_rgba(val) for val in vals]
    return colors


colors = get_colors(kdeObj(np.vstack((im_dist_vec, act_dist_vec))))
pcorr = pearsonr(im_dist_vec, act_dist_vec)
sns.regplot(x=im_dist_vec, y=act_dist_vec,
            scatter_kws={'alpha': 1, 'color': colors}, line_kws={'color': (77/255, 1, 192/255)})
# plt.scatter(pw_euclid_dist_mat[upper_tri_inds], lpips_dist_mat_full[upper_tri_inds], alpha=0.02)
plt.xlabel('image_dist')
plt.ylabel('activity_dist')
plt.title("Pearson's corr 95%%CI (%0.2f, %0.2f)" % pcorr.confidence_interval())
plt.show()
#%%
fig, axs = plt.subplots(2, 1)
axs[0].hist(im_dist_vec)
axs[1].hist(act_dist_vec)

plt.show()
#%%
# some wrappers by binxu
from core.utils.plot_utils import save_imgrid, show_imgrid, make_grid
from core.utils.montage_utils import crop_all_from_montage, crop_from_montage
# show_imgrid(all_alpha_final_ims, 16)
#%%
# if args.G == "fc6":
#     np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
#              generations=generations, scores_all=scores_all, codes_fin=codes_all[-80:, :])
# else:
#     np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
#              generations=generations, scores_all=scores_all, codes_all=codes_all)
# visualize_trajectory(scores_all, generations, codes_arr=codes_all, title_str=methodlab).savefig(
#     join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))

