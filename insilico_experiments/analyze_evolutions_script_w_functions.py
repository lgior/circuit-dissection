import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import matplotlib as mpl
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


def load_folder_evolutions(data_dir_path):
    files_list = os.listdir(data_dir_path)
    scores_npzfiles = [s for s in files_list if re.match('scores.*', s)]
    epoch_seeds = [re.search(r'.*\_(\d*)\.jpg', s).group(1) for s in files_list if re.match(r'.*\_\d*\.jpg', s)]
    final_imgfiles = [s for s in files_list if re.match(r'last.*\_\d*.*\.jpg', s)]
    traj_imgfiles = [s for s in files_list if re.match(r'best.*\_\d*\.jpg', s)]
    initcode_files = [s for s in files_list if re.match(r'init.*\_\d*\.npy', s)]

    scores_list = []
    generations_list = []
    final_img_list = []
    traj_img_list = []
    init_codes_list = []
    # TODO check that files load always in the same order, or sort them in analysis for reproducibility
    for _ifile in range(len(scores_npzfiles)):
        # Check that image and score files match.
        assert(re.search(r'scores(.*)\.npz', scores_npzfiles[_ifile]).group(1) ==
               re.search(r'lastgen(.*)_score.*', final_imgfiles[_ifile]).group(1))
        evolution_exp_data = np.load(join(data_dir_path, scores_npzfiles[_ifile]))
        scores_list.append(evolution_exp_data['scores_all'])
        generations_list.append(evolution_exp_data['generations'])
        # TODO automatically load correct number of files, check the length of the other vectors.
        if "fc6" in final_imgfiles[_ifile]:
            num_ims = 40
        else:
            num_ims = 28
        final_im_grid_array = np.asarray(Image.open(join(data_dir_path, final_imgfiles[_ifile])))
        traj_im_grid_array = np.asarray(Image.open(join(data_dir_path, traj_imgfiles[_ifile])))
        final_img_list.append(crop_all_from_montage(final_im_grid_array, imgsize=256, totalnum=num_ims))
        traj_img_list.append(crop_all_from_montage(traj_im_grid_array, imgsize=256, totalnum=num_ims))
        init_codes_list.append(np.load(join(data_dir_path, initcode_files[_ifile])))
        # it seems the image grids have all padding of 2 pixels by default.
    data_dict_list = {'scores': scores_list,
                      'generations': generations_list,
                      'final_ims': final_img_list,
                      'traj_ims': traj_img_list,
                      'init_codes': init_codes_list,
                      'scores_files': scores_npzfiles,
                      'ims_files': traj_imgfiles}
    return data_dict_list


def get_mean_ci(data_vec):
    return bootstrap((data_vec,), np.mean, confidence_level=0.95, n_resamples=10).confidence_interval


def get_exp_summary_dicts(generations_list, scores_list, epoch_traj_imgfiles):
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
    return exp_dict_list


def get_final_scores_list(generations_list, scores_list):
    final_scores_list = []
    for igen, curr_gens in enumerate(generations_list):
        score_vec = np.array(scores_list[igen])
        final_scores_list.append(score_vec[curr_gens == curr_gens[-1]])
    return final_scores_list


def make_weighted_im_grid(im_list, weights_list, file_path, nrow=None):
    # TODO test that weights are normalized to 0-1, allowed alpha values. Or add this to function.
    if nrow is None:
        nrow = np.int32(np.floor(np.sqrt(len(weights_list))))
    alpha_ims_list = []
    # weight images using the alpha channel
    for f_im, f_sc in zip(im_list, weights_list):
        _tmp_im = Image.fromarray(f_im)
        _tmp_im.putalpha(np.int32(np.floor(255 * f_sc + 10)))
        alpha_ims_list.append(ToTensor()(_tmp_im))
    # Sort images in decreasing weight order.
    alpha_ims_list = [im for _, im in sorted(zip(weights_list, alpha_ims_list), reverse=True, key=lambda pair: pair[0])]
    ims_alpha_grid = ToPILImage()(make_grid(alpha_ims_list, nrow=nrow))
    # Save as png
    # https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg
    try:
        ims_alpha_grid.save(file_path)
    except IOError:
        print('Cannot save weighted image grid to %s' % file_path)
    return


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


def compute_lpips_dist_matrix_from_im_list(im_list):
    # TODO add gpu flag
    # Set network to GPU, fails if you forget to move the scaling_layer, as lpips defaults to cpu.
    cuda0 = torch.device('cuda:0')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    loss_fn_alex = lpips.LPIPS(net='alex', version='0.1')
    loss_fn_alex.to(cuda0)
    loss_fn_alex.scaling_layer.to(cuda0)
    ims_tensor = torch.stack(list(map(ToTensor(), all_final_ims)))
    norm_ims_tensor = 2 * ims_tensor - 1

    # We want to generate a distance matrix, so let's loop over diagonals to use the batch processing or pytorch
    # alternatively use a list of pairs of indices and process it in batches.
    lpips_dist_mat = np.zeros((len(all_final_ims),) * 2)
    _decile = 0
    for idiag in range(len(all_final_ims)):
        if idiag == _decile * len(all_final_ims) // 10:
            print('\r%d/%d' % (idiag, len(all_final_ims)), end='\r')
            _decile += 1
        row_inds = np.arange(len(all_final_ims) - idiag)
        # inds_list = (list(zip(row_inds, row_inds + idiag)))
        lpips_tensor = loss_fn_alex.forward(norm_ims_tensor[row_inds, :, :, :].to(cuda0),
                                            norm_ims_tensor[row_inds + idiag, :, :, :].to(cuda0))
        lpips_dist_mat[row_inds, row_inds + idiag] = lpips_tensor.squeeze().detach().cpu().numpy()
        # if idiag == 2: break
    print('lpips done!')
    return lpips_dist_mat + lpips_dist_mat.T

#%%
# plotting stuff


def plot_scatter_line_colored_by_density(x_array, y_array, ax, scatter_kws, line_kws):
    def get_colors(vals):
        colors = np.zeros((len(vals), 3))
        norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())
        colors = [mpl.cm.ScalarMappable(norm=norm, cmap='magma_r').to_rgba(val) for val in vals]
        return colors

    kde_obj = kde(np.vstack((x_array, y_array)))
    colors = get_colors(kde_obj(np.vstack((x_array, y_array))))
    ax = sns.regplot(x=x_array, y=y_array, ax=ax, scatter_kws=scatter_kws | {'color': colors}, line_kws=line_kws)
    return ax


def plot_scatter_im_vs_act_dist(im_dist_array, act_dist_array, save_path):
    # format figure
    cm = 1/2.54
    fig = plt.figure(figsize=(15 * cm, 15 * cm))
    ax = fig.add_subplot(111)
    scatter_kws = {'alpha': 1, 's': 2}
    line_kws = {'color': (77 / 255, 1, 192 / 255)}
    plot_scatter_line_colored_by_density(im_dist_array, act_dist_array, ax, scatter_kws, line_kws)
    # fig.set_size_inches(15 * cm, 15 * cm)

    plt.rcParams['font.size'] = 14
    pcorr = pearsonr(im_dist_array, act_dist_array)
    # plt.scatter(pw_euclid_dist_mat[upper_tri_inds], lpips_dist_mat_full[upper_tri_inds], alpha=0.02)
    plt.xlabel('image distance (lpips)')
    plt.ylabel('activity distance (L2)')
    plt.title("Pearson's corr 95%%CI (%0.2f, %0.2f)" % pcorr.confidence_interval())
    plt.xlim([0, 1])
    plt.ylim(0)
    plt.savefig(save_path)
    plt.show()
    return


def make_fig_lpips_activity_matrices(lpips_dist_mat_full, pw_euclid_dist_mat, save_path):
    cm = 1 / 2.54
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15 * cm, 7 * cm))
    cbar_size = [1.05, 0.25, 0.1, 0.5]
    l_im = axs[0].imshow(lpips_dist_mat_full)
    fig.colorbar(l_im, ax=axs[0], cax=axs[0].inset_axes(cbar_size))
    act_im = axs[1].imshow(pw_euclid_dist_mat)
    fig.colorbar(act_im, ax=axs[1], cax=axs[1].inset_axes(cbar_size))

    axs[0].set_title('Image distance (lpips)')
    axs[1].set_title('Activity distance (L2)')
    fig.suptitle('Distance matrices over images')
    plt.savefig(save_path)
    plt.show()
    return


def make_fig_distance_hists(im_dist_vec, act_dist_vec, save_path):
    cm = 1/2.54
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10 * cm, 10 * cm))
    axs[0].hist(im_dist_vec)
    axs[1].hist(act_dist_vec)

    fig.suptitle('Pairwise distance distributions')
    axs[0].set_xlabel('Image distance (lpips)')
    axs[1].set_xlabel('Activity distance (L2)')
    axs[0].set_ylabel('# pairs')
    axs[1].set_ylabel('# pairs')
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.savefig(save_path)
    plt.show()
    return
#%%
torch.cuda.empty_cache()

rootdir = r"C:\Users\giordano\Documents\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
layer_str = '.classifier.Linear6'
unit_idx = 373#131#373
#%%
data_dirs = [idir for idir in next(os.walk(rootdir))[1] if re.match('alexnet-eco-.*%s_%s'%(layer_str, unit_idx), idir)]

#%%
epochList = [1, 80]
#%%
# check if there is a more efficient way to import, and verify all list are in the same order of seeds
curr_epoch = 4
epoch_pattern = 'alexnet-eco-%03d_.*' % curr_epoch
epoch_dir = next((s for s in data_dirs if re.match(epoch_pattern, s)), None)
# Create figure folder
fig_dir_path = join(rootdir, epoch_dir, 'analysis_figs')
if not os.path.exists(fig_dir_path):
    os.mkdir(fig_dir_path)

epoch_dir_path = join(rootdir, epoch_dir)

#%%
dir_data_dict = load_folder_evolutions(epoch_dir_path)
#%%
exp_dict_list = get_exp_summary_dicts(dir_data_dict['generations'], dir_data_dict['scores'], dir_data_dict['ims_files'])
final_scores_list = get_final_scores_list(dir_data_dict['generations'], dir_data_dict['scores'])
final_img_list = dir_data_dict['final_ims']
#%%
generator_names = [exp['generator'] for exp in exp_dict_list]
for final_imgs, final_scores, name in zip(final_img_list, final_scores_list, generator_names):
    final_scores_norm = (final_scores - np.min(final_scores)) / (np.max(final_scores) - np.min(final_scores))
    # weigh images by scores using the alpha channel
    file_path = join(fig_dir_path, "all_final_generations%s.png" % name)
    make_weighted_im_grid(final_imgs, final_scores_norm, file_path)

#%%
plot_activity_trajectories_ci(exp_dict_list)
#%%
# Merge list of list into single list.
all_final_ims = list(itertools.chain.from_iterable(final_img_list))
all_final_scores = list(itertools.chain.from_iterable(final_scores_list))
all_final_scores_norm = ((all_final_scores - np.min(all_final_scores)) /
                         (np.max(all_final_scores) - np.min(all_final_scores)))
#%%
file_path = join(fig_dir_path, 'all_evolutions_final_generation_weighted_grid.png')
make_weighted_im_grid(all_final_ims, all_final_scores_norm, file_path)
#%%
lpips_dist_mat_full = compute_lpips_dist_matrix_from_im_list(all_final_ims)
#%%
# compute activity distance, just the diff
# TODO check if to use normalize scores or raw ones here, normalizing whole vector doesn't affect correlations.
# pw_euclid_dist_mat = pairwise_euclidean_distance(torch.asarray(all_final_scores).unsqueeze(-1))
pw_euclid_dist_mat = pairwise_euclidean_distance(torch.asarray(all_final_scores_norm).unsqueeze(-1))

np.savez(join(rootdir, epoch_dir, 'distance_mats'), lpips_dist_mat_full, pw_euclid_dist_mat)
#%%
save_path = join(fig_dir_path, "distance_matrices.png")
make_fig_lpips_activity_matrices(lpips_dist_mat_full, pw_euclid_dist_mat, save_path)
#%%
upper_tri_inds = np.triu_indices(pw_euclid_dist_mat.shape[0], k=1)
act_dist_vec = pw_euclid_dist_mat[upper_tri_inds].numpy()
im_dist_vec = lpips_dist_mat_full[upper_tri_inds]

save_path = join(fig_dir_path, "distance_scatter_line.png")
plot_scatter_im_vs_act_dist(im_dist_vec, act_dist_vec, save_path)
save_path = join(fig_dir_path, "distance_histograms.png")
make_fig_distance_hists(im_dist_vec, act_dist_vec, save_path)
#%%
# some wrappers by binxu
# from core.utils.plot_utils import save_imgrid, show_imgrid, make_grid
# from core.utils.montage_utils import crop_all_from_montage, crop_from_montage
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

