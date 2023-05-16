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
    codes_list = []
    weights_silenced_list = []
    # TODO check that files load always in the same order, or sort them in analysis for reproducibility
    for _ifile in range(len(scores_npzfiles)):
        # Check that image and score files match.
        assert (re.search(r'scores(.*)\.npz', scores_npzfiles[_ifile]).group(1) ==
                re.search(r'lastgen(.*)_score.*', final_imgfiles[_ifile]).group(1))
        evolution_exp_data = np.load(join(data_dir_path, scores_npzfiles[_ifile]))  # this was my original version
        evolution_exp_data = np.load(join(data_dir_path, scores_npzfiles[_ifile]), allow_pickle=True)
        scores_list.append(evolution_exp_data['scores_all'])
        generations_list.append(evolution_exp_data['generations'])
        if 'codes_all' in evolution_exp_data:
            codes_list.append(evolution_exp_data['codes_all'])
        elif 'codes_fin' in evolution_exp_data:
            codes_list.append(evolution_exp_data['codes_fin'])
        if 'weights_silenced' in evolution_exp_data:
            weights_silenced_list.append(evolution_exp_data['weights_silenced'])
        else:
            weights_silenced_list.append([])
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
                      'codes': codes_list,
                      'weights_silenced': weights_silenced_list[0],  # all weights should be the same so let's save once
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
                                       else (scores_list[igen][curr_gens == step][0],) * 2
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


def plot_activity_trajectories_ci(list_of_score_dicts, save_path):
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
    plt.savefig(save_path)
    plt.show()


def compute_lpips_dist_matrix_from_im_list(im_list):
    torch.cuda.empty_cache()
    # TODO add gpu flag
    # Set network to GPU, fails if you forget to move the scaling_layer, as lpips defaults to cpu.
    cuda0 = torch.device('cuda:0')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    loss_fn_alex = lpips.LPIPS(net='alex', version='0.1')
    loss_fn_alex.to(cuda0)
    loss_fn_alex.scaling_layer.to(cuda0)
    ims_tensor = torch.stack(list(map(ToTensor(), im_list)))
    norm_ims_tensor = 2 * ims_tensor - 1

    # We want to generate a distance matrix, so let's loop over diagonals to use the batch processing or pytorch
    # alternatively use a list of pairs of indices and process it in batches.
    lpips_dist_mat = np.zeros((len(im_list),) * 2)
    _decile = 0
    for idiag in range(len(im_list)):
        if idiag == _decile * len(im_list) // 10:
            print('\r%d/%d' % (idiag, len(im_list)))
            _decile += 1
        row_inds = np.arange(len(im_list) - idiag)
        # inds_list = (list(zip(row_inds, row_inds + idiag)))
        with torch.no_grad():
            im_tensor_1 = norm_ims_tensor[row_inds, :, :, :].to(cuda0)
            im_tensor_2 = norm_ims_tensor[row_inds + idiag, :, :, :].to(cuda0)
            lpips_tensor = loss_fn_alex.forward(im_tensor_1, im_tensor_2)
        lpips_dist_mat[row_inds, row_inds + idiag] = lpips_tensor.squeeze().detach().cpu().numpy()
        # if idiag == 2: break
        del im_tensor_1, im_tensor_2
        torch.cuda.empty_cache()
    print('lpips done!')
    return lpips_dist_mat + lpips_dist_mat.T


# %%
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
    cm = 1 / 2.54
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
    plt.ylim([0, 1])  # because activity is normalized to 0-1 distance is bounded.
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
    cm = 1 / 2.54
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10 * cm, 10 * cm))
    axs[0].hist(im_dist_vec)
    axs[1].hist(act_dist_vec)

    fig.suptitle('Pairwise distance distributions')
    axs[0].set_xlabel('Image distance (lpips)')
    axs[1].set_xlabel('Activity distance (L2)')
    axs[0].set_ylabel('# pairs')
    axs[1].set_ylabel('# pairs')
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.savefig(save_path)
    plt.show()
    return


# Higher level function to process a directory


def preprocess_artificial_unit_evolution_dir(artificial_unit_dir, overwrite_existing=False):
    # Create figure folder
    fig_dir_path = join(artificial_unit_dir, 'analysis_figs')
    if not os.path.exists(fig_dir_path):
        os.mkdir(fig_dir_path)
    elif not overwrite_existing:
        print('Directory %s has already been preprocessed, '
              '\nuse overwrite_existing=True to reprocess it' % artificial_unit_dir)
        return  # the folder has been preprocessed, so don't waste more compute.

    epoch_dir_path = artificial_unit_dir
    # load data files
    dir_data_dict = load_folder_evolutions(epoch_dir_path)
    exp_dict_list = get_exp_summary_dicts(dir_data_dict['generations'], dir_data_dict['scores'],
                                          dir_data_dict['ims_files'])
    final_scores_list = get_final_scores_list(dir_data_dict['generations'], dir_data_dict['scores'])
    final_img_list = dir_data_dict['final_ims']
    # plot image grids
    generator_names = [exp['generator'] for exp in exp_dict_list]
    for final_imgs, final_scores, name in zip(final_img_list, final_scores_list, generator_names):
        final_scores_norm = (final_scores - np.min(final_scores)) / (np.max(final_scores) - np.min(final_scores))
        # weigh images by scores using the alpha channel
        file_path = join(fig_dir_path, "all_final_generations%s.png" % name)
        make_weighted_im_grid(final_imgs, final_scores_norm, file_path)

    # plot evolution trajectories
    save_path = join(fig_dir_path, "evolution_trajectories.png")
    plot_activity_trajectories_ci(exp_dict_list, save_path)

    # Merge list of list into single list.
    all_final_ims = list(itertools.chain.from_iterable(final_img_list))
    all_final_scores = list(itertools.chain.from_iterable(final_scores_list))
    all_final_scores_norm = ((all_final_scores - np.min(all_final_scores)) /
                             (np.max(all_final_scores) - np.min(all_final_scores)))
    # Plot image grid of all final evolutions
    file_path = join(fig_dir_path, 'all_evolutions_final_generation_weighted_grid.png')
    make_weighted_im_grid(all_final_ims, all_final_scores_norm, file_path)
    # Compute image distance matrix
    lpips_dist_mat_full = compute_lpips_dist_matrix_from_im_list(all_final_ims)
    # compute activity distance, just the diff
    # TODO check if to use normalize scores or raw ones here, normalizing whole vector doesn't affect correlations.
    # pw_euclid_dist_mat = pairwise_euclidean_distance(torch.asarray(all_final_scores).unsqueeze(-1))
    pw_euclid_dist_mat = pairwise_euclidean_distance(torch.asarray(all_final_scores_norm).unsqueeze(-1))
    # Save distance matrices, they take long to compute
    np.savez(join(artificial_unit_dir, 'distance_mats'), lpips_dist_mat_full, pw_euclid_dist_mat)
    # plot distance matrices
    save_path = join(fig_dir_path, "distance_matrices.png")
    make_fig_lpips_activity_matrices(lpips_dist_mat_full, pw_euclid_dist_mat, save_path)
    # plot correlation for image and activity distances for upper triangular matrix, because matrices are symmetric.
    upper_tri_inds = np.triu_indices(pw_euclid_dist_mat.shape[0], k=1)
    act_dist_vec = pw_euclid_dist_mat[upper_tri_inds].numpy()
    im_dist_vec = lpips_dist_mat_full[upper_tri_inds]

    save_path = join(fig_dir_path, "distance_scatter_line.png")
    plot_scatter_im_vs_act_dist(im_dist_vec, act_dist_vec, save_path)
    save_path = join(fig_dir_path, "distance_histograms.png")
    make_fig_distance_hists(im_dist_vec, act_dist_vec, save_path)


# %%
torch.cuda.empty_cache()

rootdir = r"C:\Users\giordano\Documents\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
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
    preprocess_artificial_unit_evolution_dir(join(rootdir, epoch_dir))

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
if 0:
    # TODO correct saving np.savez command to give names to variables x=x, y=y, defaults to arr_0, arr_1, ...
    im_dist_array = dist_mats['arr_0']
    act_dist_array = dist_mats['arr_1']
    # compute pearson correlation
    pcorr = pearsonr(im_dist_array, act_dist_array)
    # plt.scatter(pw_euclid_dist_mat[upper_tri_inds], lpips_dist_mat_full[upper_tri_inds], alpha=0.02)
    plt.xlabel('image distance (lpips)')
    plt.ylabel('activity distance (L2)')
    plt.title("Pearson's corr 95%%CI (%0.2f, %0.2f)" % pcorr.confidence_interval())

    scores_npzfiles = [s for s in files_list if re.match('scores.*', s)]
    epoch_seeds = [re.search(r'.*\_(\d*)\.jpg', s).group(1) for s in files_list if re.match(r'.*\_\d*\.jpg', s)]
    final_imgfiles = [s for s in files_list if re.match(r'last.*\_\d*.*\.jpg', s)]
    traj_imgfiles = [s for s in files_list if re.match(r'best.*\_\d*\.jpg', s)]
    initcode_files = [s for s in files_list if re.match(r'init.*\_\d*\.npy', s)]

    # Create figure folder
    fig_dir_path = join(artificial_unit_dir, 'analysis_figs')
    if not os.path.exists(fig_dir_path):
        os.mkdir(fig_dir_path)
    elif not overwrite_existing:
        1
        # break  # the folder has been preprocessed, so don't waste more compute.

    epoch_dir_path = artificial_unit_dir
    # load data files
    dir_data_dict = load_folder_evolutions(epoch_dir_path)


# %%
def parse_optim_generator_labels(best_im_file_list):
    # TODO check this works for arbitrary optimizer names as some can include multiple underscores
    init_seeds = [re.search(r'besteachgen(.*)_(\d+)\.jpg', im_file).group(2) for im_file in best_im_file_list]
    optim_labels = [re.search(r'besteachgen(.*?)(?:_.*|$)', label).group(1) for label in best_im_file_list]
    generator_labels = [re.search(r'besteachgen.*?(?:_(.*))?_\d+\.jpg', label).group(1) for label in best_im_file_list]
    # generator_labels = [re.search(r'.*?(?:_(.*)|$)', label).group(1) for label in optim_labels]
    generator_labels = [gen if gen is not None else 'BigGAN' for gen in generator_labels]  # extend if more generators
    return optim_labels, generator_labels


# %% test for the perturbation analysis
torch.cuda.empty_cache()

# Sections to load GAN
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names,
                                       save_as_images)  # install it from huggingface GR
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN


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


G_fc6 = load_GAN("fc6")
G = load_GAN("BigGAN")

rootdir = r"C:\Users\giordano\Documents\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
rootdir = r"M:\Data"  # r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
layer_str = '.classifier.Linear6'
layer_str = '.Linearfc'
unit_idx = 373  # 13, 373, 14, 12, 72, 66, 78
unit_pattern = 'alexnet-eco-080.*%s_%s' % (layer_str, unit_idx)
unit_pattern = 'resnet50_%s_%s' % (layer_str, unit_idx)
perturbation_pattern = '_kill_topFraction_'
# unit_pattern += perturbation_pattern
# %%
original_dir = [idir for idir in next(os.walk(rootdir))[1] if re.match(unit_pattern + '$', idir)]
data_dirs = [idir for idir in next(os.walk(rootdir))[1] if re.match(unit_pattern + perturbation_pattern, idir)]


# %%

def extrac_top_scores_and_images_from_dir(artificial_unit_dir, overwrite_existing=False):
    """Extract the top scores and image from each evolution in the given directory
    returns:
     list of scores
     tensor of size num_evolutions x 3 x 256 x 256 for the images used for fc6 and biggan"""
    data_dict_list = load_folder_evolutions(data_dir_path=artificial_unit_dir)

    optim_labels, generator_labels = parse_optim_generator_labels(data_dict_list['ims_files'])
    data_dict_list['generator_labels'], data_dict_list['optim_labels'] = generator_labels, optim_labels

    # fc6 as used now relies on 40 images per iteration, BiGGAN on 28, this relates to the optimization algorithm and
    # dimensions of the latent vectors
    # fc6 codes are only saved for last 80 images, so last 2 generations
    max_inds, max_codes, max_scores = [], [], []
    for i_exp, scores in enumerate(data_dict_list['scores']):
        # # TODO make this more flexible depending on the number of stored codes for fc6
        # if data_dict_list['generator_labels'][i_exp] == 'fc6':
        #     scores = scores[-80:]
        scores = scores[-len(data_dict_list['codes'][i_exp]):]
        # get the bestimage of last 2 generations for fc6 or all for biggan
        max_ind = np.argmax(scores)
        max_codes.append(data_dict_list['codes'][i_exp][max_ind])
        max_scores.append(scores[max_ind])
        max_inds.append(max_ind)

    biggan_inds = [index for (index, item) in enumerate(data_dict_list['generator_labels']) if item == "BigGAN"]
    fc6_inds = [index for (index, item) in enumerate(data_dict_list['generator_labels']) if item == "fc6"]
    biggan_codes = [max_codes[index] for index in biggan_inds]
    fc6_codes = [max_codes[index] for index in fc6_inds]

    biggan_ims = G.visualize(torch.from_numpy(np.array(biggan_codes)).float().cuda()).cpu()
    fc6_ims = G_fc6.visualize(torch.from_numpy(np.array(fc6_codes)).float().cuda()).cpu()

    # put images back into a list and sort by original ordering
    unsorted_inds = biggan_inds + fc6_inds
    original_inds = sorted(range(len(unsorted_inds)), key=unsorted_inds.__getitem__)
    max_im_tensor = torch.cat((biggan_ims, fc6_ims))[original_inds, ...]
    # # max_image_list = [ToPILImage()(im) for im in torch.cat((biggan_ims, fc6_ims))]
    # # max_image_list = [max_image_list[ind] for ind in original_inds]
    # max_image_list = [ToPILImage()(im) for im in max_im_tensor]  # maybe could be used for other purposes
    return max_scores, max_im_tensor


# %%

def save_images_manipulated_by_scores(score: list, im_tensor: torch.Tensor, fig_dir_path: str,
                                      artificial_unit_dir: str):
    # TODO change variable names to sth more generic
    unsorted_grid = ToPILImage()(make_grid(im_tensor, nrow=5))
    unsorted_grid.save(join(fig_dir_path, "unsorted_grid__%s.jpg" % (os.path.split(artificial_unit_dir)[1])))
    unsorted_grid.show()

    # weigthed average
    weights = torch.tensor(score).float()
    weighted_mean_image = ToPILImage()((weights.reshape((-1, 1, 1, 1)) * im_tensor).sum(axis=0) / weights.sum())
    # weighted_mean_image = ToPILImage()((weights.reshape((-1, 1, 1, 1)) * im_tensor).mean(axis=0))
    weighted_mean_image.save(join(fig_dir_path, "weighted_mean__%s.jpg" % (os.path.split(artificial_unit_dir)[1])))
    weighted_mean_image.show()

    # average
    unweighted_mean_image = ToPILImage()(im_tensor.mean(axis=0))
    unweighted_mean_image.save(join(fig_dir_path, "unweighted_mean__%s.jpg" % (os.path.split(artificial_unit_dir)[1])))
    unweighted_mean_image.show()

    descending_sort_inds = np.argsort(score)[::-1].copy()  # copy to avoid negative stride error in pytorch
    sorted_grid = ToPILImage()(make_grid(im_tensor[descending_sort_inds, ...], nrow=5))
    sorted_grid.save(join(fig_dir_path, "sorted_grid__%s.jpg" % (os.path.split(artificial_unit_dir)[1])))
    sorted_grid.show()


def save_top_images_from_dir(artificial_unit_dir, overwrite_existing=False):
    # Create figure folder
    fig_dir_path = join(artificial_unit_dir, 'analysis_figs')
    if not os.path.exists(fig_dir_path):
        os.mkdir(fig_dir_path)
    elif not overwrite_existing:
        print('Directory %s has already been preprocessed, '
              '\nuse overwrite_existing=True to reprocess it' % artificial_unit_dir)
        return

    max_scores, max_im_tensor = extrac_top_scores_and_images_from_dir(artificial_unit_dir)
    save_images_manipulated_by_scores(max_scores, max_im_tensor, fig_dir_path, artificial_unit_dir)
    return


# #%%
# mtg_exp = ToPILImage()(make_grid(biggan_ims, nrow=4))
# mtg_exp.show()
# mtg_exp = ToPILImage()(make_grid(fc6_ims, nrow=4))
# mtg_exp.show()
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
    max_scores, max_im_tensor = extrac_top_scores_and_images_from_dir(artificial_unit_dir)
    max_scores_list.append(max_scores)
    max_im_tensor_list.append(max_im_tensor)

# %%
colors = ['k' if tip == 'abs' else 'b' if tip == 'exc' else 'r' for tip in perturbation_data_dict['type']]
fig, ax = plt.subplots(5, len(colors) // 5, sharex=True, sharey=True)
for ii, (ax, score, color) in enumerate(zip(ax.flatten(), max_scores_list, colors)):
    ax.hist(score, alpha=0.4, color=color)
    ticks = [ii % 6 == 0, ii > 24]
    ax.tick_params(left=ticks[0], bottom=ticks[1])
plt.show()

# %%
# plot the amount of silencing versus the maximum scores
import pandas as pd
import seaborn as sns

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
n_types = len(pd.unique(max_per_condition_df.sort_values(['type', 'strength']).type))
max_n_ims_type = max_per_condition_df.sort_values(['type', 'strength']).groupby('type').apply(len).max()
for ii, (name, group) in enumerate(max_per_condition_df.sort_values(['type', 'strength']).groupby('type')):
    # print(group)
    print(name)
    for jj, row in enumerate(group.iterrows()):
        print(row)
        plt.subplot(n_types, max_n_ims_type, max_n_ims_type * ii + jj + 1)
        plt.imshow(ToPILImage()(max_per_condition_im_list[df.old_index[row[1].max_index]]))  # index is the 0th column of the row.
plt.show()
#%%
# Plot the mean image or top image
from mpl_toolkits.axes_grid1 import ImageGrid


plot_weighted_im_mean = True

fig = plt.figure(figsize=(20., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_types, max_n_ims_type),  # creates 2x2 grid of axes
                 axes_pad=0.4,  # pad between axes in inch.
                 )

max_df = df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength'], group_keys=False).apply(
    lambda group: group.sort_values('scores', ascending=False).head(1))  # set group_keys=False to avoid multiindex creation

# for ii, (name, group) in enumerate(max_per_condition_df.sort_values(['type', 'strength']).groupby('type')):
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


def list_argsort(a_list, reverse=False):
    return sorted(range(len(a_list)), key=lambda i: a_list[i], reverse=reverse)


def get_top_n_im_grid(scores: list, images: torch.Tensor, top_n: int):
    top_inds = list_argsort(scores, reverse=True)[:top_n]
    top_n_im_grid = ToPILImage()(make_grid(images[top_inds, ...], nrow=int(np.ceil(np.sqrt(top_n)))))
    return top_n_im_grid


for i_cond, (scores, images) in enumerate(zip(perturbation_data_dict['scores'], perturbation_data_dict['images'])):
    # print(scores)
    # print(list_argsort(scores, reverse=True)[:9])
    # print([scores[i] for i in list_argsort(scores, reverse=True)[:9]])
    # select top9 activating images, and convert to image grid
    top9_im_grid = get_top_n_im_grid(scores, images, top_n=9)
top9_im_grid.show()

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
        top9_im_grid = get_top_n_im_grid(scores=perturbation_data_dict['scores'][list_index],
                                         images=perturbation_data_dict['images'][list_index], top_n=9)
        plt.imshow(top9_im_grid)  # index is the 0th column of the row.
        plt.title('strength: {},\n score:{:0.2f}'.format(row_data.strength, row_data.scores))
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

# %%
# [ToPILImage()(im) for im in max_per_condition_list]
# %%
max_df = df.groupby(['type', 'strength']).max().reset_index()
max_df = df.groupby(['type', 'strength']).apply(
    max)  # this keeps the columns of type and strength, the other option makes a multiindex
# max_df = df.groupby(['type', 'strength']).max()  # max_df.index.get_level_values(1)
# sns.barplot(max_df, x='strength', y='scores', hue='type')
sns.catplot(max_df, x='strength', y='scores', hue='type', jitter=False)
plt.show()


# %%


def jitter(values, scale):
    return values + np.random.normal(0, scale, values.shape)


# fig, ax = plt.subplots(figsize=(8,6))
# df.groupby('type').plot(x='strength', y='scores', kind='scatter', ax=ax)
ax_line = sns.lineplot(data=df, x='strength', y='scores', hue='type', errorbar=('ci', 95), markers='o')
ax_scatter = sns.scatterplot(data=df, x=jitter(df.strength, 0.008), y='scores', hue='type', alpha=0.5)
# sns.stripplot(df, x='strength', y='scores', hue='type', alpha=0.5) # this makes x axis categorical
# sns.violinplot(df, x='strength', y='scores', hue='type', inner='points')
# sns.violinplot(data=df, x="strength", y="scores", inner="points", hue='type')
# sns.violinplot(data=df[df.type == 'abs'], x="strength", y="scores", inner="points")
# sns.violinplot(data=df, x='strength', y="scores", hue='type', inner="points", face=0.9)
h_leg_scatter, label_scatter = ax_scatter.get_legend_handles_labels()
plt.legend(handles=list(zip(h_leg_scatter[:-4], h_leg_scatter[-4:])), labels=label_scatter[:-4], loc='lower left')
# plt.legend()
plt.show()

# TODO plot the strength of the input with respect to the activations
# %%
for ii, (scores, color) in enumerate(zip(max_scores_list, colors)):
    plt.scatter(np.repeat(perturbation_data_dict['strength'][ii], len(scores)), scores, alpha=0.4, color=color)
plt.show()
# %%

artificial_unit_dir = join(rootdir, data_dirs[6])
max_scores, max_im_tensor = extrac_top_scores_and_images_from_dir(artificial_unit_dir)
# save_top_images_from_dir(artificial_unit_dir, overwrite_existing=True)

# %%
import itertools

all_scores = list(itertools.chain.from_iterable(data_dict_list['scores']))
all_max_scores = [max(scores) for scores in data_dict_list['scores']]
last_gen_scores = list(itertools.chain.from_iterable([scores[-40:] for scores in data_dict_list['scores']]))
# %%

for i_exp, scores in enumerate(data_dict_list['scores']):
    if data_dict_list['generator_labels'][i_exp] == 'fc6':
        color = 'violet'
    else:
        color = 'lightgreen'
    x = np.sort(scores)
    score_cdf = np.cumsum(x) / x.sum()
    plt.step(x, score_cdf, color=color)
x = np.sort(all_scores)
score_cdf = np.cumsum(x) / x.sum()
plt.step(x, score_cdf, color='black')
plt.show()
# %%
plt.hist(all_scores, alpha=0.4, label='all', density=True)
plt.hist(all_max_scores, alpha=0.4, label='max', density=True)
plt.hist(last_gen_scores, alpha=0.4, label='last', density=True)
plt.legend()
plt.show()

# %%

# %%
imgs = G.visualize(torch.from_numpy(np.array(fc6_codes)).float().cuda()).cpu()

mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
# mtg_exp.save(join(savedir, "besteachgen%s_%05d.jpg" % (methodlab, RND,)))
mtg = ToPILImage()(make_grid(imgs, nrow=7))
# mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
