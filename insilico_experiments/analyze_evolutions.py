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
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names,
                                       save_as_images)  # install it from huggingface GR
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN


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
        # Check that image and score files match, and print an error if they don't
        try:
            assert (re.search(r'scores(.*)\.npz', scores_npzfiles[_ifile]).group(1) ==
                    re.search(r'lastgen(.*)_score.*', final_imgfiles[_ifile]).group(1)), \
                'ERROR: scores and final images do not match: %s | %s' % \
                (scores_npzfiles[_ifile], final_imgfiles[_ifile])
        except AssertionError as error:
            print(error)

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


def plot_scatter_line_colored_by_density(x_array, y_array, ax, scatter_kws=None, line_kws=None, cmap=None, compute_kde=True):
    def get_colors(vals):
        colors = np.zeros((len(vals), 3))
        norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())
        # cmap = sns.color_palette("flare", as_cmap=True)
        colors = [mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(val) for val in vals]
        return colors

    if cmap is None:
        cmap = mpl.colormaps['magma_r']
    # set default scatter and line kws
    scatter_kws = scatter_kws or {'alpha': 1}
    line_kws = line_kws or {'color': 'xkcd:charcoal'}

    if compute_kde:
        kde_obj = kde(np.vstack((x_array, y_array)))
        colors = get_colors(kde_obj(np.vstack((x_array, y_array))))
    else:
        colors = 'xkcd:dusty lavender'
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

# SECTION made for analyzing silencing exps
def parse_optim_generator_labels(best_im_file_list):
    # TODO check this works for arbitrary optimizer names as some can include multiple underscores
    init_seeds = [re.search(r'besteachgen(.*)_(\d+)\.jpg', im_file).group(2) for im_file in best_im_file_list]
    optim_labels = [re.search(r'besteachgen(.*?)(?:_.*|$)', label).group(1) for label in best_im_file_list]
    generator_labels = [re.search(r'besteachgen.*?(?:_(.*))?_\d+\.jpg', label).group(1) for label in best_im_file_list]
    # generator_labels = [re.search(r'.*?(?:_(.*)|$)', label).group(1) for label in optim_labels]
    generator_labels = [gen if gen is not None else 'BigGAN' for gen in generator_labels]  # extend if more generators
    return optim_labels, generator_labels


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


    # put images back into a list and sort by original ordering
    unsorted_inds = biggan_inds + fc6_inds
    original_inds = sorted(range(len(unsorted_inds)), key=unsorted_inds.__getitem__)
    if len(biggan_inds) >= 1 and len(fc6_inds) >= 1:
        print('Loading %d BigGAN images and %d fc6 images' % (len(biggan_inds), len(fc6_inds)))
        biggan_ims = G.visualize(torch.from_numpy(np.array(biggan_codes)).float().cuda()).cpu()
        fc6_ims = G_fc6.visualize(torch.from_numpy(np.array(fc6_codes)).float().cuda()).cpu()
        max_im_tensor = torch.cat((biggan_ims, fc6_ims))[original_inds, ...]
    elif len(biggan_inds) >= 1:
        print('Loading %d BigGAN images' % len(biggan_inds))
        biggan_ims = G.visualize(torch.from_numpy(np.array(biggan_codes)).float().cuda()).cpu()
        max_im_tensor = biggan_ims[original_inds, ...]
    elif len(fc6_inds) >= 1:
        fc6_ims = G_fc6.visualize(torch.from_numpy(np.array(fc6_codes)).float().cuda()).cpu()
        max_im_tensor = fc6_ims[original_inds, ...]

    # # max_image_list = [ToPILImage()(im) for im in torch.cat((biggan_ims, fc6_ims))]
    # # max_image_list = [max_image_list[ind] for ind in original_inds]
    # max_image_list = [ToPILImage()(im) for im in max_im_tensor]  # maybe could be used for other purposes
    # make a vector indicating which images are fc6 and which are biggan
    max_generator_labels = [data_dict_list['generator_labels'][ind] for ind in original_inds]
    # make it an optional output
    return max_scores, max_im_tensor, max_generator_labels

# %%

def save_images_manipulated_by_scores(score: list, im_tensor: torch.Tensor, fig_dir_path: str,
                                      artificial_unit_dir: str):
    # TODO change variable names to sth more generic
    unsorted_grid = ToPILImage()(make_grid(im_tensor, nrow=5))
    unsorted_grid.save(join(fig_dir_path, "unsorted_grid__%s.jpg" % (os.path.split(artificial_unit_dir)[1])))
    unsorted_grid.show()

    # weigthed average # TODO rescale image to 0-1 range to avoid artifacts
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


def list_argsort(a_list, reverse=False):
    return sorted(range(len(a_list)), key=lambda i: a_list[i], reverse=reverse)


def get_top_n_im_grid(scores: list, images: torch.Tensor, top_n: int):
    top_inds = list_argsort(scores, reverse=True)[:top_n]
    top_n_im_grid = ToPILImage()(make_grid(images[top_inds, ...], nrow=int(np.ceil(np.sqrt(top_n)))))
    return top_n_im_grid


def jitter(values, scale):
    return values + np.random.normal(0, scale, values.shape)


def get_recorded_unit_indices(rootdir, net_str, layer_str):
    # "C:\Users\gio\Data\resnet50_linf8_.Linearfc_701_kill_topFraction_in_weight_-0.30"
    unit_pattern = net_str + '_' + layer_str  # match
    pattern = re.compile(rf'{re.escape(unit_pattern)}_(\d+)(_.*)?')
    unit_indices = [int(pattern.findall(idir)[0][0]) for idir in next(os.walk(rootdir))[1] if pattern.search(idir)]
    # unique unit_indices without numpy
    unit_indices = list(dict.fromkeys(unit_indices))
    return unit_indices


def get_recorded_layers_from_network(rootdir, net_str):
    pattern = re.compile(rf'{re.escape(net_str)}_(.*)_\d+(_.*)?')
    print(pattern)
    layer_str = [pattern.findall(idir)[0][0] for idir in next(os.walk(rootdir))[1] if pattern.search(idir)]
    layer_str = list(dict.fromkeys(layer_str))
    return layer_str


def get_unit_data_dirs(rootdir, net_str, layer_str, unit_idx, experiment_pattern):
    unit_pattern = net_str + '_' + layer_str + '_' + str(unit_idx)
    data_dirs = [tmp_dir for tmp_dir in next(os.walk(rootdir))[1]
                 if re.match(unit_pattern + '$', tmp_dir) or re.match(unit_pattern + experiment_pattern, tmp_dir)]

    return data_dirs


def find_directories_matching_pattern(rootdir, pattern):
    matching_dirs = []
    for _, dir_names, _ in os.walk(rootdir):
        matching_dirs.extend([tmp_dir for tmp_dir in dir_names if re.match(pattern, tmp_dir)])
    return matching_dirs




def get_complete_experiment_units(rootdir, net_str, layer_str, experiment_pattern, experiment_re_str, full_experiment_suffix_list):
    experiment_re = re.compile(experiment_re_str)  # re.compile(r'.*(_kill.*)$')
    unit_indices = get_recorded_unit_indices(rootdir, net_str, layer_str)
    complete_experiment_units = {}
    for unit_idx in unit_indices:
        data_dirs = get_unit_data_dirs(rootdir, net_str, layer_str, unit_idx, experiment_pattern)
        # get a list from the perturbation suffixes _kill.*$

        unit_experiment_suffix_list = [re.search(experiment_re, data_dir).group(1) for data_dir in data_dirs
                                       if experiment_pattern in data_dir]
        # check that all experiments are present
        if not set(full_experiment_suffix_list).issubset(set(unit_experiment_suffix_list)):
            print(f'Unit {unit_idx} is missing experiments: '
                  f'{set(full_experiment_suffix_list).difference(set(unit_experiment_suffix_list))}')
            complete_experiment_units[unit_idx] = set(full_experiment_suffix_list).difference(set(unit_experiment_suffix_list))
        else:
            complete_experiment_units[unit_idx] = None
    # print the list of keys for which the value is none as the complete unit indices
    print('Complete unit indices: ', [key for key, value in complete_experiment_units.items() if value is None])
    return complete_experiment_units


def get_complete_experiment_list(exc_inh_weights=None, abs_weights=None):
    # complete list of experiments
    full_experiment_suffixes = {}
    # fill with a list comprehension
    # print as strings with two decimals 0.2f
    if abs_weights is None:
        abs_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .99]
    full_experiment_suffixes['kill_topFraction_abs_in_weight'] = [
        f'_kill_topFraction_abs_in_weight_{x:.2f}'for x in abs_weights
    ]
    if exc_inh_weights is None:
        exc_inh_weights = list(np.round(np.arange(0.1, 1.1, 0.1), 2))
    exc_experiment_suffixes = [f'_kill_topFraction_in_weight_{x:.2f}'for x in exc_inh_weights]
    inh_experiment_suffixes = [f'_kill_topFraction_in_weight_{-x:.2f}'for x in exc_inh_weights]
    full_experiment_suffixes['kill_topFraction_in_weight'] = inh_experiment_suffixes + exc_experiment_suffixes

    # full_experiment_suffixes = [
    #     '_kill_topFraction_abs_in_weight_0.10',
    #     '_kill_topFraction_abs_in_weight_0.20',
    #     '_kill_topFraction_abs_in_weight_0.30',
    #     '_kill_topFraction_abs_in_weight_0.40',
    #     '_kill_topFraction_abs_in_weight_0.50',
    #     '_kill_topFraction_abs_in_weight_0.60',
    #     '_kill_topFraction_abs_in_weight_0.70',
    #     '_kill_topFraction_abs_in_weight_0.80',
    #     '_kill_topFraction_abs_in_weight_0.90',
    #     '_kill_topFraction_abs_in_weight_0.99',
    #     '_kill_topFraction_in_weight_-0.10',
    #     '_kill_topFraction_in_weight_-0.20',
    #     '_kill_topFraction_in_weight_-0.30',
    #     '_kill_topFraction_in_weight_-0.40',
    #     '_kill_topFraction_in_weight_-0.50',
    #     '_kill_topFraction_in_weight_-0.60',
    #     '_kill_topFraction_in_weight_-0.70',
    #     '_kill_topFraction_in_weight_-0.80',
    #     '_kill_topFraction_in_weight_-0.90',
    #     '_kill_topFraction_in_weight_-1.00',
    #     '_kill_topFraction_in_weight_0.10',
    #     '_kill_topFraction_in_weight_0.20',
    #     '_kill_topFraction_in_weight_0.30',
    #     '_kill_topFraction_in_weight_0.40',
    #     '_kill_topFraction_in_weight_0.50',
    #     '_kill_topFraction_in_weight_0.60',
    #     '_kill_topFraction_in_weight_0.70',
    #     '_kill_topFraction_in_weight_0.80',
    #     '_kill_topFraction_in_weight_0.90',
    #     '_kill_topFraction_in_weight_1.00'
    # ]

    return full_experiment_suffixes

