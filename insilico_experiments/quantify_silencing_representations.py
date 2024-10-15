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
import pickle
from functools import reduce
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from tqdm import tqdm
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


def get_perturbation_data_dict(rootdir, unit_data_dirs):
    perturbation_data_dict = {key: [] for key in ['type', 'strength', 'scores', 'images', 'generator']}

    for data_dir in unit_data_dirs:
        if 'kill' in data_dir:
            perturbation_fraction = float(re.search(r'.*_([\-\.\d]+)?', data_dir).group(1))
            perturbation_type = 'abs' if 'abs' in data_dir else 'exc' if perturbation_fraction >= 0 else 'inh'
        else:
            perturbation_fraction = 0
            perturbation_type = 'none'
        perturbation_data_dict['strength'].append(abs(perturbation_fraction))
        perturbation_data_dict['type'].append(perturbation_type)

        max_scores, max_im_tensor, max_im_generator = anevo.extrac_top_scores_and_images_from_dir(join(rootdir, data_dir))
        print("%s has %d scores" % (data_dir, len(max_scores)))

        perturbation_data_dict['scores'].append(max_scores)
        perturbation_data_dict['images'].append(max_im_tensor)
        perturbation_data_dict['generator'].append(max_im_generator)
    return perturbation_data_dict


#%%

def preprocess_dataframe(df, unit_scores_str):
    df_copy = df.copy()  # Create a copy of the dataframe
    if unit_scores_str in df_copy.columns:
        df_copy.drop(columns=[unit_scores_str], inplace=True)  # Modify the copy
    df_copy['type'] = df_copy['type'].astype(str)
    return df_copy  # Return the modified copy

def compute_mean_cosine_similarity(silenced_output, control_output):
    # check if silenced_output and control_output are equal
    if torch.equal(silenced_output, control_output):
        cosine_similarity = pairwise_cosine_similarity(silenced_output, control_output)
        # get upper triangular values
        triu_inds = torch.triu(torch.ones_like(cosine_similarity), diagonal=1)
        cosine_similarity = cosine_similarity[triu_inds == 1]
    else:
        cosine_similarity = pairwise_cosine_similarity(silenced_output, control_output)
    return torch.mean(cosine_similarity).detach().cpu().numpy()

def compute_mean_euclidean_norm(silenced_output, control_output):
    euclidean_norm = torch.norm(silenced_output, dim=1)
    # return mean along samples
    return torch.mean(euclidean_norm).detach().cpu().numpy()

def compute_rsa_values(df, image_list, scorer, control_output, similarity_metric='cosine_similarity'):
    rsa_values = []
    for index, row in df.iterrows():
        silenced_images_tensor = image_list[index].to(torch.device('cuda:0'))
        silenced_output = scorer.model(silenced_images_tensor)
        silenced_output = torch.nn.functional.softmax(silenced_output, dim=1)
        if similarity_metric == 'cosine_similarity':
            mean_cosine_similarity = compute_mean_cosine_similarity(silenced_output, control_output)
        elif similarity_metric == 'euclidean_norm':
            mean_cosine_similarity = compute_mean_euclidean_norm(silenced_output, control_output)
        rsa_values.append(mean_cosine_similarity)
    return rsa_values


def preprocess_and_compute_rsa_df(df_dict, image_dict, network_list, complete_unit_index_list,
                                  net_str, layer_str, preprocess_data_to_file=True, preprocessed_data_dir_path=None,
                                  similarity_metric='cosine_similarity'):

    file_name = f'rsa_{net_str}_{layer_str}_metric_{similarity_metric}.pkl'
    file_path = join(preprocessed_data_dir_path, file_name)
    #check if the file exists
    if os.path.exists(file_path) and preprocess_data_to_file:
        df_rsa = pd.read_pickle(file_path)
        return df_rsa

    # Copy the dictionaries to avoid modifying the originals
    df_dict_copy = df_dict.copy()
    image_dict_copy = image_dict.copy()
    # Initialize the DataFrame structure using the first network and unit
    first_unit = complete_unit_index_list[0]
    df_first_unit = df_dict_copy[first_unit]
    df_rsa = initialize_df_structure(df_first_unit, network_list, complete_unit_index_list)

    # Preprocess the data before the loop
    for unit, df in df_dict.items():
        unit_scores_str = f'scores_{unit}'
        df_dict_copy[unit] = preprocess_dataframe(df, unit_scores_str)

    # Iterate over each network
    for net in network_list:
        scorer = TorchScorer(net)

        # Call function to compute RSA values
        for unit in complete_unit_index_list:
            df = df_dict_copy[unit]
            image_list = image_dict_copy[unit]
            control_tensor = image_list[df['type'].eq('none').idxmax()].to(torch.device('cuda:0'))
            control_output = torch.nn.functional.softmax(scorer.model(control_tensor), dim=1)

            # Call function to add column and update DataFrame
            update_df_with_rsa(df_rsa, df, unit, net,
                               compute_rsa_values(df, image_list, scorer, control_output,
                                                  similarity_metric=similarity_metric))

    # save the dataframe
    if preprocess_data_to_file:
        df_rsa.to_pickle(file_path)

    return df_rsa


def update_df_with_rsa(df_rsa, df, unit, net, rsa_values):
    # Add a new column for the current network if not already present
    if (unit, net) not in df_rsa.columns:
        df_rsa[unit, net] = np.nan
    df_rsa.loc[:, (unit, net)] = rsa_values


def initialize_df_structure(df, network_list, complete_unit_index_list):
    df_copy = df.copy().reset_index().set_index(['index', 'type', 'strength'])
    net_columns = pd.MultiIndex.from_product([complete_unit_index_list, network_list], names=['unit', 'test_network'])
    return df_copy.reindex(columns=net_columns)


#%%
import logging
logging.basicConfig(level=logging.INFO)
# even less logging
# logging.getLogger().setLevel(logging.WARNING)

def preprocess_silencing_data_to_df(rootdir, net_str, layer_str, complete_unit_index_list,
                                    preprocess_data_to_file=True,
                                    overwrite=False,
                                    perturbation_pattern=None, preprocessed_data_dir_path=None):
    """
    Preprocess silencing data into DataFrames and save them to files.

    Args:
    - rootdir (str): Root directory containing the silencing data.
    - net_str (str): Network name.
    - layer_str (str): Layer name.
    - complete_unit_index_list (list): List of unit indices.
    - preprocess_data_to_file (bool): Flag indicating whether to preprocess data and save to files (default: True).
    - perturbation_pattern (str): Pattern for identifying experiment directories (default: None).
    - preprocessed_data_dir_path (str): Directory path to save preprocessed data files (default: None).

    Returns:
    - df_dict (dict): Dictionary containing DataFrames.
    - image_dict (dict): Dictionary containing image data.
    """
    columns = ['type', 'strength', 'scores', 'generator']  # 'images' is too big to be stored in a dataframe
    df_dict = {}
    image_dict = {}

    file_name_df = f'{net_str}_{layer_str}_dataframes_dict.pkl'
    file_name_images = f'{net_str}_{layer_str}_images_dict.pkl'
    file_path_df = join(preprocessed_data_dir_path, file_name_df)
    file_path_images = join(preprocessed_data_dir_path, file_name_images)


    # Check if preprocessed data already exists
    if not overwrite:
        if os.path.exists(file_path_df) and os.path.exists(file_path_images):
            # print("Preprocessed data already exists. Loading from files...")
            logging.info("Preprocessed data already exists. Loading from files...")
            with open(file_path_df, 'rb') as f_df, open(file_path_images, 'rb') as f_images:
                df_dict = pickle.load(f_df)
                image_dict = pickle.load(f_images)
            return df_dict, image_dict

    for unit in tqdm(complete_unit_index_list, desc="Processing units"):
        # Get the directories of all the experiments for a given unit
        unit_data_dirs = anevo.get_unit_data_dirs(rootdir, net_str, layer_str, unit,
                                                  experiment_pattern=perturbation_pattern)
        # Extract the top image and score per evolution experiment inside a folder
        perturbation_data_dict = get_perturbation_data_dict(rootdir, unit_data_dirs)
        df = pd.DataFrame.from_dict({key: perturbation_data_dict[key] for key in columns})
        df.reset_index(names='list_index', inplace=True)
        # Rename column 'scores' to 'scores_unit'
        unit_scores_str = f'scores_{unit}'
        df.rename(columns={'scores': unit_scores_str}, inplace=True)
        df_dict[unit] = df
        image_dict[unit] = perturbation_data_dict['images']

    # Save DataFrames and image data to files
    if preprocess_data_to_file:
        try:
            # Save DataFrames to file
            with open(file_path_df, 'wb') as f_df:
                pickle.dump(df_dict, f_df)
            logging.info(f"Saved {file_name_df}")

            # Save image data to file
            with open(file_path_images, 'wb') as f_images:
                pickle.dump(image_dict, f_images)
            logging.info(f"Saved {file_name_images}")

        except Exception as e:
            logging.error(f"Error occurred while saving preprocessed data: {e}")


    return df_dict, image_dict
#%%

figures_dir = os.path.join('C:', 'Users', 'gio', 'Data', 'figures', 'silencing')
# os.makedirs(figures_dir, exist_ok=True)
# make dir if it doesn't exist
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# TODO refactor following repeated code into another function or method
rootdir = r"C:\Users\gio\Data"  # since my home folder got full

# check if a folder named preprocessed_data exists, if not create it
preprocessed_data_dir_path = join(rootdir, 'preprocessed_data')
if not os.path.exists(preprocessed_data_dir_path):
    os.makedirs(preprocessed_data_dir_path)

net_str = 'vgg16'
# net_str = 'alexnet'  # 'resnet50', 'alexnet-eco-080'
# net_str = 'resnet50'  # 'resnet50', 'alexnet-eco-080'
# net_str = 'resnet50_linf0.5'
# net_str = 'resnet50_linf1'
# net_str = 'resnet50_linf2'
# net_str = 'resnet50_linf4'
# net_str = 'resnet50_linf8'
# net_str = 'alexnet-single-neuron2'
# net_str = 'alexnet-single-neuron_Caos-12192023-005'
# net_str = 'alexnet-single-neuron_Caos-12202023-003'
# net_str = 'alexnet-single-neuron_Caos-01162024-010'
# net_str = 'alexnet-single-neuron_Caos-01172024-006'
# net_str = 'alexnet-single-neuron_Caos-01182024-005'
# net_str = 'alexnet-single-neuron_Caos-01252024-009'
# net_str = 'alexnet-single-neuron_Caos-02082024-005'
# net_str = 'alexnet-single-neuron_Caos-02092024-006'
# net_str = 'alexnet-single-neuron_Caos-02132024-007'
# net_str = 'alexnet-single-neuron_Caos-02152024-006'
# net_str = 'alexnet-single-neuron_Caos-02202024-007'
# net_str = 'alexnet-single-neuron_Caos-02212024-005'
# net_str = 'alexnet-single-neuron_Caos-02222024-005'
# net_str = 'alexnet-single-neuron_Caos-02272024-005'
# net_str = 'alexnet-single-neuron_Caos-02292024-007'
# net_str = 'alexnet-single-neuron_Caos-03052024-005'
# net_str = 'alexnet-single-neuron_Caos-03082024-002'
# net_str = 'alexnet-single-neuron_Caos-03112024-003'
# net_str = 'alexnet-single-neuron_Caos-03122024-002'
# net_str = 'alexnet-single-neuron_Caos-03132024-002'
# net_str = 'alexnet-single-neuron_Caos-03142024-003'
# net_str = 'alexnet-single-neuron_Caos-03182024-002'
# net_str = 'alexnet-single-neuron_Caos-03202024-003'
# net_str = 'alexnet-single-neuron_Caos-03212024-002'
# net_str = 'alexnet-single-neuron_Caos-03222024-002'
# net_str = 'alexnet-single-neuron_Caos-03252024-002'
# net_str = 'alexnet-single-neuron_Caos-03262024-002'
# net_str = 'alexnet-single-neuron_Caos-03292024-002'
# net_str = 'alexnet-single-neuron_Caos-04012024-002'
# net_str = 'alexnet-single-neuron_Caos-04022024-002'
# net_str = 'alexnet-single-neuron_Caos-04052024-002'
# net_str = 'alexnet-single-neuron_Caos-04092024-002'
# net_str = 'alexnet-single-neuron_Caos-04102024-002'
# net_str = 'alexnet-single-neuron_Caos-04112024-002'
# net_str = 'alexnet-single-neuron_Diablito-12042024-002'
# net_str = 'alexnet-single-neuron_Caos-04162024-002' # control bad fit
# net_str = 'alexnet-single-neuron_Caos-04172024-002' # control bad fit
# net_str = 'alexnet-single-neuron_Diablito-19042024-002'
# net_str = 'alexnet-single-neuron_Diablito-22042024-002'
# net_str = 'alexnet-single-neuron_Diablito-24042024-002'
# net_str = 'alexnet-single-neuron_Diablito-25042024-002'
# net_str = 'alexnet-single-neuron_Caos-04302024-002'
# net_str = 'alexnet-single-neuron_Diablito-07052024-002'
# net_str = 'alexnet-single-neuron_Caos-05092024-002'
# net_str = 'alexnet-single-neuron_Caos-05132024-002'
# net_str = 'alexnet-single-neuron_Caos-05142024-003'
# net_str = 'alexnet-single-neuron_Diablito-16052024-002'
# net_str = 'alexnet-single-neuron_Diablito-07062024-002'
# net_str = 'alexnet-single-neuron_Diablito-17062024-002'
# net_str = 'alexnet-single-neuron_Diablito-18062024-002'
# net_str = 'alexnet-single-neuron_Diablito-19062024-003'
# net_str = 'alexnet-single-neuron_Diablito-20062024-002'
# net_str = 'alexnet-single-neuron_Diablito-21062024-002'
# net_str = 'alexnet-single-neuron_Diablito-24062024-002'
# net_str = 'alexnet-single-neuron_Diablito-25062024-002'
# net_str = 'alexnet-single-neuron_Diablito-26062024-002'
# net_str = 'alexnet-single-neuron_Caos-06272024-002'
# net_str = 'alexnet-single-neuron_Diablito-08072024-002'
# net_str = 'alexnet-single-neuron_Diablito-09072024-002'
# net_str = 'alexnet-single-neuron_Diablito-11072024-002'
# net_str = 'alexnet-single-neuron_Diablito-12072024-002'
# net_str = 'alexnet-single-neuron_Diablito-15072024-002'
# net_str = 'alexnet-single-neuron_Diablito-16072024-002'
# net_str = 'alexnet-single-neuron_Diablito-17072024-002'
# net_str = 'alexnet-single-neuron_Diablito-18072024-002'
# net_str = 'alexnet-single-neuron_Diablito-19072024-002'
# net_str = 'alexnet-single-neuron_Diablito-22072024-002'
# net_str = 'alexnet-single-neuron_Caos-07232024-006'
# net_str = 'alexnet-single-neuron_Diablito-24072024-002'
# net_str = 'alexnet-single-neuron_Diablito-25072024-002'
# net_str = 'alexnet-single-neuron_Diablito-26072024-002'
# net_str = 'alexnet-single-neuron_Diablito-29072024-002'
net_str = 'alexnet-single-neuron_Diablito-30072024-002'

experiment_class = 'imagenette'
experiment_class = 'neurons'

if any(s in net_str for s in ['alexnet-eco', 'resnet50']):
    layer_str = '.Linearfc'  # for resnet50 and alexnet-eco-080
else:
    layer_str = '.classifier.Linear6'  # for alexnet

print(f'Possible units to analyze: {anevo.get_recorded_unit_indices(rootdir, net_str, layer_str)}')

perturbation_regex = r'.*(_kill.*)$'
perturbation_pattern = '_kill_topFraction_'

full_experiment_suffixes = anevo.get_complete_experiment_list()
if experiment_class == 'imagenette':
    # this one is for full experiments
    full_experiment_suffixes = anevo.get_complete_experiment_list(exc_inh_weights=[0.2, 0.4, 0.6, 0.8, 1.0],
                                                                  abs_weights=[0.2, 0.4, 0.6, 0.8, 0.99])
elif experiment_class == 'neurons':
    full_experiment_suffixes = anevo.get_complete_experiment_list(exc_inh_weights=[0.25, 0.5, 0.75, 1.0],
                                                                  abs_weights=[0.25, 0.5, 0.75, 0.99])

silencing_experiment_suffixes = []
for suffix in full_experiment_suffixes:
    if 'kill' in suffix:
        silencing_experiment_suffixes.extend(full_experiment_suffixes[suffix])


complete_unit_indices = anevo.get_complete_experiment_units(
    rootdir, net_str, layer_str, perturbation_pattern, perturbation_regex, silencing_experiment_suffixes
)
complete_unit_index_list = [key for key, value in complete_unit_indices.items() if value is None]

imagenette_units = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

if experiment_class == 'imagenette':
    assert set(imagenette_units + [373]).issubset(set(complete_unit_index_list))

#%%
import zlib

compile_all_units = True
# compile_all_units = False

net_str_list = [
    # 'alexnet-single-neuron_Caos-12192023-005',
    'alexnet-single-neuron_Caos-12202023-003',
    'alexnet-single-neuron_Caos-01162024-010',
    'alexnet-single-neuron_Caos-01172024-006',
    'alexnet-single-neuron_Caos-01182024-005',
    'alexnet-single-neuron_Caos-01252024-009',
    'alexnet-single-neuron_Caos-02082024-005',
    'alexnet-single-neuron_Caos-02092024-006',
    'alexnet-single-neuron_Caos-02132024-007',
    'alexnet-single-neuron_Caos-02152024-006',
    'alexnet-single-neuron_Caos-02202024-007',
    'alexnet-single-neuron_Caos-02212024-005',
    'alexnet-single-neuron_Caos-02222024-005',
    'alexnet-single-neuron_Caos-02272024-005',
    'alexnet-single-neuron_Caos-02292024-007',
    'alexnet-single-neuron_Caos-03052024-005',
    'alexnet-single-neuron_Caos-03082024-002',
    'alexnet-single-neuron_Caos-03112024-003',
    'alexnet-single-neuron_Caos-03122024-002',
    'alexnet-single-neuron_Caos-03132024-002',
    'alexnet-single-neuron_Caos-03142024-003',
    'alexnet-single-neuron_Caos-03182024-002',
    'alexnet-single-neuron_Caos-03202024-003',
    'alexnet-single-neuron_Caos-03212024-002',
    'alexnet-single-neuron_Caos-03222024-002',
    'alexnet-single-neuron_Caos-03252024-002',
    'alexnet-single-neuron_Caos-03262024-002',
    'alexnet-single-neuron_Caos-03292024-002',
    'alexnet-single-neuron_Caos-04012024-002',
    'alexnet-single-neuron_Caos-04022024-002',
    'alexnet-single-neuron_Caos-04052024-002',
    'alexnet-single-neuron_Caos-04092024-002',
    'alexnet-single-neuron_Caos-04102024-002',
    'alexnet-single-neuron_Caos-04112024-002',
    'alexnet-single-neuron_Diablito-12042024-002',
    # 'alexnet-single-neuron_Caos-04162024-002', # control bad fit
    # 'alexnet-single-neuron_Caos-04172024-002', # control bad fit
    'alexnet-single-neuron_Diablito-19042024-002',
    'alexnet-single-neuron_Diablito-22042024-002',
    'alexnet-single-neuron_Diablito-24042024-002',
    'alexnet-single-neuron_Diablito-25042024-002',
    'alexnet-single-neuron_Caos-04302024-002',
    'alexnet-single-neuron_Diablito-07052024-002',
    'alexnet-single-neuron_Caos-05092024-002',
    'alexnet-single-neuron_Caos-05132024-002',
    'alexnet-single-neuron_Caos-05142024-003',
    'alexnet-single-neuron_Diablito-16052024-002',
    'alexnet-single-neuron_Diablito-07062024-002',
    'alexnet-single-neuron_Diablito-17062024-002',
    'alexnet-single-neuron_Diablito-18062024-002',
    'alexnet-single-neuron_Diablito-19062024-003',
    'alexnet-single-neuron_Diablito-20062024-002',
    'alexnet-single-neuron_Diablito-21062024-002',
    'alexnet-single-neuron_Diablito-24062024-002',
    'alexnet-single-neuron_Diablito-25062024-002',
    'alexnet-single-neuron_Diablito-26062024-002',
    'alexnet-single-neuron_Caos-06272024-002',
    'alexnet-single-neuron_Diablito-08072024-002',
    'alexnet-single-neuron_Diablito-09072024-002',
    'alexnet-single-neuron_Diablito-11072024-002',
    'alexnet-single-neuron_Diablito-12072024-002',
    'alexnet-single-neuron_Diablito-15072024-002',
    'alexnet-single-neuron_Diablito-16072024-002',
    'alexnet-single-neuron_Diablito-17072024-002',
    'alexnet-single-neuron_Diablito-18072024-002',
    'alexnet-single-neuron_Diablito-19072024-002',
    'alexnet-single-neuron_Diablito-22072024-002',
    'alexnet-single-neuron_Caos-07232024-006',
    'alexnet-single-neuron_Diablito-24072024-002',
    'alexnet-single-neuron_Diablito-25072024-002',
    'alexnet-single-neuron_Diablito-26072024-002',
    'alexnet-single-neuron_Diablito-29072024-002',
    'alexnet-single-neuron_Diablito-30072024-002'
]

net_to_exclude = [
    'alexnet-single-neuron_Caos-04162024-002', # control bad fit
    'alexnet-single-neuron_Caos-04172024-002', # control bad fit
]

if compile_all_units:

    crc32_hashes = [zlib.crc32(s.encode()) & 0xFFFFFFFF for s in net_str_list]
    # this is the zero-th unit for the neuron models, as we use one network per neuron so far, they could be combined in the future
    complete_unit_index_list = [0]
    df_dict = {}
    image_dict = {}
    # Iterate over complete_unit_index_list, making a dictionary with the hash as key for joining all neuron models
    for net_str_index, tmp_net_str in tqdm(zip(crc32_hashes, net_str_list)):
        complete_unit_index = complete_unit_index_list[0]
        df_dict_tmp, image_dict_tmp = preprocess_silencing_data_to_df(rootdir, tmp_net_str, layer_str, complete_unit_index_list,
                                                              preprocess_data_to_file=True,
                                                              perturbation_pattern=perturbation_pattern,
                                                              preprocessed_data_dir_path=preprocessed_data_dir_path)

        print(f'Processing {tmp_net_str} {layer_str} {complete_unit_index}')
        # Rename the column in df_dict_tmp[complete_unit_index] to scores_(net_str_index)
        df_dict_tmp[complete_unit_index]['scores_' + str(net_str_index)] = df_dict_tmp[complete_unit_index].pop('scores_' + str(complete_unit_index))

        # Add the new dictionary to a master dictionary with the net_str_index as key
        df_dict[net_str_index] = df_dict_tmp[complete_unit_index]
        image_dict[net_str_index] = image_dict_tmp[complete_unit_index]


    complete_unit_index_list = crc32_hashes
    # index of network to exclude
    net_to_exclude_index = [i for i, net in enumerate(net_str_list) if net in net_to_exclude]
    units_to_exclude = [complete_unit_index_list[i] for i in net_to_exclude_index]

else:
    preprocess_data_to_file = True

    # Compile the single unit data into a dictionary of dataframes and images
    df_dict, image_dict = preprocess_silencing_data_to_df(rootdir, net_str, layer_str, complete_unit_index_list,
                                                          preprocess_data_to_file=True,
                                                          perturbation_pattern=perturbation_pattern,
                                                          preprocessed_data_dir_path=preprocessed_data_dir_path)


    # load dataframes and images from file
    file_name = f'{net_str}_{layer_str}_dataframes_dict.pkl'
    with open(join(preprocessed_data_dir_path, file_name), 'rb') as f:
        df_dict = pickle.load(f)


#%%
def plot_scores_vs_silencing(df, net_str, layer_str, unit, ax=None, params_dict=None):

    with plt.rc_context(params_dict):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        # Plot the strength of response vs strength of input perturbation
        # fig, ax = plt.subplots(figsize=(8,6))
        # df.groupby('type').plot(x='strength', y='scores', kind='scatter', ax=ax)
        # set the current axis to ax
        plt.sca(ax)
        score_name = [column for column in df.columns if 'score' in column][0]
        ax_line = sns.lineplot(data=df, x='strength', y=score_name, hue='type', errorbar=('ci', 95), markers='o',
                               hue_order=['abs', 'inh', 'exc', 'none'],
                               palette=['tab:purple', 'tab:orange', 'tab:green', 'tab:blue'],
                               lw=4, err_kws={'edgecolor': None})
        ax_scatter = sns.scatterplot(data=df, x=anevo.jitter(df.strength, 0.008), y=score_name, hue='type', alpha=0.5,
                                     hue_order=['abs', 'inh', 'exc', 'none'],
                                     palette=['tab:purple', 'tab:orange', 'tab:green', 'tab:blue'],
                                     ec='w', s=50)
        # sns.stripplot(df, x='strength', y='scores', hue='type', alpha=0.5) # this makes x axis categorical
        # sns.violinplot(df, x='strength', y='scores', hue='type', inner='points')
        # sns.violinplot(data=df, x="strength", y="scores", inner="points", hue='type')
        # sns.violinplot(data=df[df.type == 'abs'], x="strength", y="scores", inner="points")
        # sns.violinplot(data=df, x='strength', y="scores", hue='type', inner="points", face=0.9)
        h_leg_scatter, label_scatter = ax_scatter.get_legend_handles_labels()
        plt.legend(handles=list(zip(h_leg_scatter[:-4], h_leg_scatter[-4:])), labels=label_scatter[:-4],
                   labelspacing=0.2,  handlelength=1,
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.legend()

        plt.title(f'{net_str} {layer_str} \n unit {unit}')
        # ax.set_title('Testing', font=fpath)
        # prop = fm.FontProperties(fname=f.name)
        # ax.set_title('this is a special font:\n%s' % github_url, fontproperties=prop)
        plt.xlabel('Silencing strength')
        plt.ylabel('Response (a.u.)')
        # plt.ylim([0, 1])
        # ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        # ax.set_yticks([0, 0.5, 1])
        # get current figure
        # fig.tight_layout()
        # plt.show()

#%%
params = {
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 14,
    'figure.titlesize': 14,
    'figure.labelsize': 14,
    'font.family': 'Arial',
    'legend.fontsize': 10,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': False
}
# make a grid of 4x3 axes for plotting, one for each unit
with plt.rc_context(params):
    # TODO make plotting adaptable to the number of units
    nrows= 3 if experiment_class == 'imagenette' else np.ceil(np.sqrt(len(complete_unit_index_list))).astype(int)
    ncols=np.ceil(len(complete_unit_index_list) / nrows).astype(int)
    if compile_all_units or experiment_class == 'imagenette':
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2.5, nrows*2), dpi=300, sharex=True, sharey=False)
    else:
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), dpi=300, sharex=True, sharey=True)

for ii, unit in enumerate(complete_unit_index_list):
    # now preprocess the dataframe to extract maximum scores and images
    df = df_dict[unit]
    unit_scores_str = f'scores_{unit}'
    if 'generator' in df.columns:
        df = df.explode([unit_scores_str, 'generator'])
    else:
        df = df.explode([unit_scores_str])
    df[unit_scores_str] = df[unit_scores_str].astype("float")
    df.type = df.type.astype("string")
    # df = df.reset_index(names='list_index') ## changed to inplace in loading function
    # with plt.rc_context(params):
    if compile_all_units:
        net_str = 'neurons'
        unit = net_str_list[crc32_hashes.index(unit)].split('_')[-1]
    plot_scores_vs_silencing(df, net_str, layer_str, unit, ax=axes[ii // ncols, ii % ncols], params_dict=params)
    if ii != 0:
        axes[ii // ncols, ii % ncols].get_legend().remove()
    # remove axis labels for akk but bottom left plot
    if ii // ncols != nrows - 1 or ii % ncols != 0:
        axes[ii // ncols, ii % ncols].set_xlabel('')
        axes[ii // ncols, ii % ncols].set_ylabel('')

fig.tight_layout()
# save figure
if compile_all_units:
    fig_name = f'all_units_{layer_str}_scores_vs_silencing_strength.png'
    fig.savefig(join(figures_dir, fig_name), dpi=300)
    fig_name = f'all_units_{layer_str}_scores_vs_silencing_strength.pdf'
    fig.savefig(join(figures_dir, fig_name), dpi=300)
else:
    fig_name = f'{net_str}_{layer_str}_scores_vs_silencing_strength.png'
    fig.savefig(join(figures_dir, fig_name), dpi=300)
    fig_name = f'{net_str}_{layer_str}_scores_vs_silencing_strength.pdf'
    fig.savefig(join(figures_dir, fig_name), dpi=300)
plt.show()


#%%


# file_name = f'{net_str}_{layer_str}_images_dict.pkl'
# # file_name = f'{net_str}_{layer_str}_image_dict.pkl'
# with open(join(preprocessed_data_dir_path, file_name), 'rb') as f:
#     image_dict = pickle.load(f)


#%%

network_list = ['alexnet',
                'resnet50', 'resnet50_linf0.5', 'resnet50_linf1', 'resnet50_linf2', 'resnet50_linf4', 'resnet50_linf8']

similarity_metric = 'cosine_similarity'
# similarity_metric = 'euclidean_norm'
# Save the DataFrame if preprocessing to file is enabled
if compile_all_units:
    # file_name = f'rsa_all_neurons_{layer_str}_metric_{similarity_metric}_neurips.pkl'
    file_name = f'rsa_all_neurons_{layer_str}_metric_{similarity_metric}_iclr.pkl'
    if os.path.exists(join(preprocessed_data_dir_path, file_name)):
        df_rsa = pd.read_pickle(join(preprocessed_data_dir_path, file_name))
    else:
        # todo here we can loop over the already saved single unit dataframes
        preprocess_data_to_file = False
        df_rsa = preprocess_and_compute_rsa_df(df_dict, image_dict, network_list, complete_unit_index_list,
                                               net_str, layer_str, preprocess_data_to_file, preprocessed_data_dir_path,
                                               similarity_metric=similarity_metric)
        df_rsa.to_pickle(join(preprocessed_data_dir_path, file_name))
else:
    df_rsa = preprocess_and_compute_rsa_df(df_dict, image_dict, network_list, complete_unit_index_list,
                                           net_str, layer_str, preprocess_data_to_file, preprocessed_data_dir_path,
                                           similarity_metric=similarity_metric)



#%%
# recover a dataframe with the mean cosine similarity for each type and strength in a tabular format
df_rsa_table = df_rsa.stack().stack().reset_index(name=f'mean_{similarity_metric}')
# average mean_cosine_similarity over the test_network level when grouping by type, strength and unit
df_rsa_table = df_rsa_table.groupby(['type', 'strength', 'unit'])[f'mean_{similarity_metric}'].mean().reset_index()


# drop units in units_to_exclude
# df_rsa_table = df_rsa_table[~df_rsa_table.unit.isin(units_to_exclude)]
#%%
# divide the mean_cosine_similarity by the control mean_cosine_similarity
for group_id, group_data in df_rsa_table.groupby(['unit']):
    control_mean_similarity = group_data[group_data.type == 'none'][f'mean_{similarity_metric}'].values.mean()
    df_rsa_table.loc[df_rsa_table.unit == group_id, f'norm_mean_{similarity_metric}'] = \
        df_rsa_table.loc[df_rsa_table.unit == group_id, f'mean_{similarity_metric}'] / control_mean_similarity

    # df_rsa_table.loc[df_rsa_table.unit == unit, 'mean_cosine_similarity'] /= df_rsa_table[df_rsa_table.type == 'none'].mean_cosine_similarity.values.mean()


#%%
# subtract the control mean_cosine_similarity from the mean_cosine_similarity
for group_id, group_data in df_rsa_table.groupby(['unit']):
    control_mean_similarity = group_data[group_data.type == 'none'][f'mean_{similarity_metric}'].values.mean()
    df_rsa_table.loc[df_rsa_table.unit == group_id, f'norm_mean_{similarity_metric}'] = \
        df_rsa_table.loc[df_rsa_table.unit == group_id, f'mean_{similarity_metric}'] - control_mean_similarity

#%%
# new for ICLR 2025
abs_df = df_rsa_table[df_rsa_table['type'] == 'none'].replace('none', 'abs')
inh_df = df_rsa_table[df_rsa_table['type'] == 'none'].replace('none', 'inh')
exc_df = df_rsa_table[df_rsa_table['type'] == 'none'].replace('none', 'exc')

# Concatenate the filtered DataFrames
df_rsa_table = pd.concat([df_rsa_table, abs_df, inh_df, exc_df], ignore_index=True)

#%%
if compile_all_units:
    df_rsa_table['experiment'] = df_rsa_table['unit'].apply(lambda x: net_str_list[crc32_hashes.index(int(x))])

#%%

# malke the previous cell into a function
def plot_rsa(ax, df_rsa_table, net_str, layer_str, y_var='mean_cosine_similarity', type_to_plot='all'):
    # add the docstring
    """
    Plot the rsa results in a figure
    :param ax: axis to plot on
    :param df_rsa_table: dataframe with the rsa results
    :param net_str: network name
    :param layer_str: layer name
    :param y_var: variable to plot on the y axis, either 'mean_cosine_similarity' or 'norm_mean_cosine_similarity'
    :param type_to_plot: type of silencing to plot, either 'all', 'inh', 'exc' or 'abs'
    :return:
    """
    # set the plotting axis to ax
    plt.sca(ax)
    control_similarities = df_rsa_table[df_rsa_table.type == 'none'][y_var].values
    # then compute the confidence interval
    ci = sns.utils.ci(sns.algorithms.bootstrap(control_similarities, n_boot=1000, func=np.mean), which=95, axis=0)
    # plot the mean and confidence interval, without edgecolor
    plt.fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
    plt.axhline(y=control_similarities.mean(), color='k', linestyle='-', alpha=0.1, lw=3)
    # plot the cosine similarity vs the silencing strength separately for each type, using seaborn
    if type_to_plot == 'all':
        sns.lineplot(data=df_rsa_table, x='strength', y=y_var, hue='type', alpha=0.5, lw=3,
                     hue_order=['abs', 'inh', 'exc', 'none'],
                     palette=['tab:purple', 'tab:orange', 'tab:green', 'tab:blue'])
    if type_to_plot == 'no_abs':
        sns.lineplot(data=df_rsa_table[df_rsa_table.type != 'abs'], x='strength', y=y_var, hue='type', alpha=0.5, lw=3,
                     hue_order=['inh', 'exc', 'none'],
                    palette=['tab:orange', 'tab:green', 'tab:blue'])
    else:
        sns.scatterplot(data=df_rsa_table[df_rsa_table.type == 'none'], x='strength', y=y_var, hue='unit',
                        ax=ax, palette='tab20', alpha=1, legend=False)
        sns.lineplot(data=df_rsa_table[df_rsa_table.type == type_to_plot], x='strength', y=y_var, hue='unit',
                     ax=ax, palette='tab20', alpha=1, lw=3)
    if y_var == 'norm_mean_cosine_similarity':
        ax.set_ylabel('normalized cosine similarity\n(control vs silencing)')
    elif y_var == 'mean_cosine_similarity':
        ax.set_ylabel('cosine similarity\n(control vs silencing)')
    elif y_var == 'mean_euclidean_norm':
        ax.set_ylabel('euclidean norm\n(control vs silencing)')
    elif y_var == 'norm_mean_euclidean_norm':
        ax.set_ylabel('normalized euclidean norm\n(control vs silencing)')
    plt.xlabel('silencing strength')
    plt.title(f'{net_str} {layer_str}')
    # square axes, with unequal aspect ratio
    plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    # move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='type' if type_to_plot == 'all' else 'unit')
    # plt.tight_layout()
    # plt.show()

#%%
import matplotlib.gridspec as gridspec
# make a figure
# fig = plt.figure(figsize=(20, 8)) # without legends is (20, 8)
fig = plt.figure(figsize=(20, 8)) # without legends is (20, 8)
# make 8 axes in a 2x4 grid
gs = gridspec.GridSpec(2, 4, figure=fig)


params = {
   'axes.labelsize': 16,
   'axes.titlesize': 20,
   'font.size': 20,
   'font.family': 'Arial',
   'legend.fontsize': 12,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'text.usetex': False,
   'figure.figsize': [21, 7]
   }


if similarity_metric == 'cosine_similarity':
    y_var = 'mean_cosine_similarity'
    y_var_norm = 'norm_mean_cosine_similarity'
elif similarity_metric == 'euclidean_norm':
    y_var = 'mean_euclidean_norm'
    y_var_norm = 'norm_mean_euclidean_norm'

var_names_list = [y_var, y_var_norm]

with plt.rc_context(params):
    # for itype, type in enumerate(['all', 'inh', 'exc', 'abs']):
    for itype, type in enumerate(['no_abs', 'inh', 'exc', 'abs']):
        for jnorm, yvar in enumerate(var_names_list):
            ax_tmp = fig.add_subplot(gs[jnorm, itype])
            plot_rsa(ax_tmp, df_rsa_table, net_str, layer_str, y_var=yvar, type_to_plot=type)
            ax_tmp.set_xticks([0, 0.5, 1])
            # set y ticks to 3 ticks maximum with locator_params
            ax_tmp.locator_params(axis='y', nbins=3)
            if itype == 1 or itype == 2:
                # remove legend from the last plot
                ax_tmp.legend().remove()
                # legend title to empty string
            if type != 'all':
                ax_tmp.set_title(type)
            # for biological neurons too many labels
            ax_tmp.legend().remove()

    plt.suptitle(f'{net_str} {layer_str}')
    plt.tight_layout()
plt.show()

if compile_all_units:
    fig_name = f'all_units_{layer_str}_rsa_vs_silencing_strength_metric_{similarity_metric}.png'
    fig.savefig(os.path.join(figures_dir, fig_name), dpi=300, bbox_inches='tight')
    fig_name = f'all_units_{layer_str}_rsa_vs_silencing_metric_{similarity_metric}.pdf'
    fig.savefig(os.path.join(figures_dir, fig_name), dpi=300, bbox_inches='tight')
else:
    fig_name = f'{net_str}_{layer_str}_rsa_vs_silencing_strength_metric_{similarity_metric}.png'
    fig.savefig(os.path.join(figures_dir, fig_name), dpi=300, bbox_inches='tight')
    fig_name = f'{net_str}_{layer_str}_rsa_vs_silencing_metric_{similarity_metric}.pdf'
    fig.savefig(os.path.join(figures_dir, fig_name), dpi=300, bbox_inches='tight')
plt.show()


if compile_all_units:
    # stop here
    raise SystemExit
#%%
# To get top scoring images, but maybe more efficient to just use the lists and index back as below
# top_9_df = df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength']).apply(lambda group: group.sort_values('scores', ascending=False).head(9))

from mpl_toolkits.axes_grid1 import ImageGrid


def plot_topn_imgrid_from_df(df, image_list, layer_str, net_str, unit, n_types, max_n_ims_type, n_top=9):
    # fig = plt.figure(figsize=(21., 6.))
    ncols = max_n_ims_type + 1
    nrows = n_types - 1

    n_images_per_condition = df.generator.apply(len).mean()
    n_im_row = int(np.ceil(np.sqrt(n_top)))
    # size per subplot
    subplot_size = 2
    if n_images_per_condition < n_top:
        fig_size = (ncols * subplot_size * int(n_images_per_condition / n_im_row), nrows * subplot_size *0.8)
    else:
        fig_size = (ncols * subplot_size, nrows * subplot_size)
    print(f' fig_size: {fig_size}')
    fig = plt.figure(figsize=fig_size, dpi=300)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.35,  # pad between axes in inch.
                     )


    for ii, (name, group) in enumerate(
            df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength']).head(1).groupby('type')):
        print(name)
        # print(group)
        # print(group.index)
        # print(group.scores)
        # print(group.old_index)
        for jj, (_, row_data) in enumerate(group.iterrows()):
            # print(jj)
            if name == 'none':
                ax = grid[ncols]
            else:
                ax = grid[ncols * ii + jj + 1]
            # print("row_data:\n", row_data)
            plt.sca(ax)
            list_index = row_data.old_index

            # top9_im_grid = anevo.get_top_n_im_grid(scores=perturbation_data_dict['scores'][list_index],
            #                                        images=perturbation_data_dict['images'][list_index], top_n=n_top)
            top9_im_grid = anevo.get_top_n_im_grid(scores=df[df.old_index == list_index].scores.tolist(),
                                                   images=image_list[list_index], top_n=n_top)
            plt.imshow(top9_im_grid)  # index is the 0th column of the row.
            if name == 'none':  # ii == 0 and jj == 0:
                plt.title('(strength: {},\nscore: {:0.2f})'.format(row_data.strength, row_data.scores))
            else:
                plt.title('({}, {:0.2f})'.format(row_data.strength, row_data.scores))

            if jj == 0 or name == 'none':
                # ax.set_ylabel(name)
                # add text to the left of the image
                ax.text(-0.1, 0.5, name, horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, rotation=90, fontdict=None)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                plt.axis('off')
                grid[0].set_visible(False)
                grid[ncols * (nrows - 1)].set_visible(False)
            plt.axis('off')
            grid[0].set_visible(False)
            if n_types == 4:
                grid[ncols * (nrows - 1)].set_visible(False)

    # fig.suptitle(f'{net_str} {layer_str}')
    fig.suptitle(f'{net_str} \n {layer_str} unit {unit}')

    fig.tight_layout()
    plt.tight_layout()
    fig_filename = os.path.join(figures_dir, f'{net_str}_{layer_str}_{unit}_preferred_top{n_top}_images.pdf')
    plt.savefig(fig_filename,
                dpi=300, format='pdf', metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, transparent=True
               )
    plt.show()


# import save_images
from torchvision.utils import save_image


params = {
   'axes.labelsize': 14,
   'axes.titlesize': 14, #16,
   'figure.titlesize': 16,
   'font.size': 18,#22,
   'font.family': 'Arial',
   'legend.fontsize': 16,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'text.usetex': False,
   'figure.figsize': [21, 7]
   }
# rcParams.update(params)

for key, value in df_dict.items():
    print(key)
    unit = key
    if compile_all_units:
        net_str = net_str_list[crc32_hashes.index(key)].split('_')[-1]
    df_temp = df_dict[key].copy()
    # iclr 2025
    df_temp = df_temp[df_temp['strength'].isin([0, 1])]

    # find the column that contains score substring in its name
    score_col = [col for col in df_temp.columns if 'scores' in col][0]
    df_temp = df_temp.explode(score_col)
    # rename scores_0 to scores
    df_temp = df_temp.rename(columns={score_col: 'scores'})
    df_temp = df_temp.reset_index(names='old_index')
    n_types = len(pd.unique(df_temp.sort_values(['type', 'strength']).type))
    n_top = df_temp.sort_values(['type', 'strength']).groupby(['type', 'strength']).apply(len).max()
    # max_n_ims_type is the number of silencing strengths for the type with the most silencing strengths
    max_n_ims_type = df_temp.groupby(['type', 'strength']).count().reset_index().groupby('type')['strength'].nunique().max()
    n_types = len(df_temp['type'].unique())

    # plot_topn_imgrid_from_df(df_temp, image_dict[0], n_types=4, max_n_ims_type=10, n_top=9)

    with mpl.rc_context(params):
        plot_topn_imgrid_from_df(df_temp, image_dict[key], layer_str, net_str, unit, n_types=n_types,
                                 max_n_ims_type=max_n_ims_type, n_top=min(1, int(np.sqrt(n_top))**2))
    break
    image_list = image_dict[key]
    top_n = 9
    for ii, (name, group) in enumerate(
            df_temp.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength']).head(1).groupby('type')):
        print(name)
        # print(group)
        # print(group.index)
        # print(group.scores)
        # print(group.old_index)
        for jj, (_, row_data) in enumerate(group.iterrows()):
            # print(jj)
            filename = os.path.join(figures_dir, f'{net_str}_{layer_str}_{unit}_preferred_top{top_n}_images.pdf')
            # print("row_data:\n", row_data)
            list_index = row_data.old_index

            # top9_im_grid = anevo.get_top_n_im_grid(scores=perturbation_data_dict['scores'][list_index],
            #                                        images=perturbation_data_dict['images'][list_index], top_n=n_top)
            top_inds = anevo.list_argsort(df_temp[df_temp.old_index == list_index].scores.tolist(), reverse=True)[:top_n]
            images = image_list[list_index][top_inds, ...]
            # save images to file, one file per image
            exp_images_dir = os.path.join(figures_dir, f'{net_str}_{layer_str}_{unit}_preferred_top{top_n}')
            os.makedirs(exp_images_dir, exist_ok=True)
            for i, image in enumerate(images):
                image_filename = os.path.join(exp_images_dir, f'{net_str}_{layer_str}_{unit}_preferred_top{top_n}'
                                                           f'_{name}{row_data.strength}_image{i}.png')
                # uncomment to save images
                save_image(image, image_filename)
            # top9_im_grid = anevo.get_top_n_im_grid(scores=df_temp[df_temp.old_index == list_index].scores.tolist(),
            #                                        images=image_list[list_index], top_n=n_top)
#%%
dooodosiidi
####################### Section to compute and plot the RSA results for all CNNS ############################


def get_complete_unit_list(net_str, rootdir):
    if any(s in net_str for s in ['alexnet-eco', 'resnet50']):
        layer_str = '.Linearfc'  # for resnet50 and alexnet-eco-080
    else:
        layer_str = '.classifier.Linear6'  # for alexnet

    print(f'Possible units to analyze: {anevo.get_recorded_unit_indices(rootdir, net_str, layer_str)}')

    perturbation_regex = r'.*(_kill.*)$'
    perturbation_pattern = '_kill_topFraction_'

    full_experiment_suffixes = anevo.get_complete_experiment_list()

    full_experiment_suffixes = anevo.get_complete_experiment_list(exc_inh_weights=[0.2, 0.4, 0.6, 0.8, 1.0],
                                                                  abs_weights=[0.2, 0.4, 0.6, 0.8, 0.99])

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

    return complete_unit_index_list, layer_str, perturbation_pattern

def load_fclayer_data_dicts_from_net_str(net_str, rootdir, preprocessed_data_dir_path):

    complete_unit_index_list, layer_str, perturbation_pattern = get_complete_unit_list(net_str, rootdir)

    # Compile the single unit data into a dictionary of dataframes and images
    df_dict, image_dict = preprocess_silencing_data_to_df(rootdir, net_str, layer_str, complete_unit_index_list,
                                                          preprocess_data_to_file=True,
                                                          perturbation_pattern=perturbation_pattern,
                                                          preprocessed_data_dir_path=preprocessed_data_dir_path)


    # load dataframes and images from file
    file_name = f'{net_str}_{layer_str}_dataframes_dict.pkl'
    with open(join(preprocessed_data_dir_path, file_name), 'rb') as f:
        df_dict = pickle.load(f)

    return df_dict, image_dict, complete_unit_index_list, layer_str


def get_mean_score_df(net_str, rootdir, preprocessed_data_dir_path):
    df_dict, image_dict, complete_unit_index_list, layer_str = load_fclayer_data_dicts_from_net_str(net_str, rootdir, preprocessed_data_dir_path)

    df_all_units = pd.DataFrame()
    for unit, df in df_dict.items():
        # copy df to avoid modifying the original
        df = df.copy()
        unit_scores_str = f'scores_{unit}'
        df['unit'] = unit
        # mean of the scores, set as np array
        df[unit_scores_str] = df[unit_scores_str].apply(lambda x: np.mean(x))
        # rename the column to scores
        df.rename(columns={unit_scores_str: 'scores'}, inplace=True)

        if df_all_units.empty:
            df_all_units = df
        else:
            df_all_units = pd.concat([df_all_units, df], axis=0)

    return df_all_units


figures_dir = os.path.join('C:', 'Users', 'gio', 'Data', 'figures', 'silencing')
# os.makedirs(figures_dir, exist_ok=True)
# make dir if it doesn't exist
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)


preprocessed_data_dir_path = join(rootdir, 'preprocessed_data')

net_str = 'resnet50'
list_net_str = ['alexnet', 'vgg16', 'resnet50', 'resnet50_linf0.5', 'resnet50_linf1', 'resnet50_linf2', 'resnet50_linf4', 'resnet50_linf8']

experiment_class = 'imagenette'
experiment_class = 'neurons'


all_net_mean_scores = pd.DataFrame()

for net_str in list_net_str:
    df_all_units = get_mean_score_df(net_str, rootdir, preprocessed_data_dir_path)
    abs_df = df_all_units[df_all_units['type'] == 'none'].replace('none', 'abs')
    inh_df = df_all_units[df_all_units['type'] == 'none'].replace('none', 'inh')
    exc_df = df_all_units[df_all_units['type'] == 'none'].replace('none', 'exc')

    # Concatenate the filtered DataFrames
    df_all_units = pd.concat([df_all_units, abs_df, inh_df, exc_df], ignore_index=True)

    # new column net_str
    df_all_units['net_str'] = net_str
    if all_net_mean_scores.empty:
        all_net_mean_scores = df_all_units
    else:
        all_net_mean_scores = pd.concat([all_net_mean_scores, df_all_units], axis=0)



#%%

rcParams = {'font.size': 16,
            'axes.labelsize': 16,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'font.family': 'arial',
            }

with plt.rc_context(rcParams):
    # facet grid plot
    g = sns.FacetGrid(all_net_mean_scores, col='net_str', hue='type', col_wrap=4, height=2.5, aspect=3/4,
                      hue_order=['abs', 'inh', 'exc', 'none'],
                      palette=['tab:purple', 'tab:orange', 'tab:green', 'tab:blue'])
    g.map(sns.lineplot, 'strength', 'scores', alpha=0.5, lw=3)
    # g.map(sns.scatterplot, 'strength', 'scores', alpha=0.5, s=50)
    # map also the confidence interval for the control type for each network
    for ax, net in zip(g.axes.flatten(), list_net_str):
        control_similarities = all_net_mean_scores[(all_net_mean_scores.type == 'none') & (all_net_mean_scores.net_str == net)].scores.values
        ci = sns.utils.ci(sns.algorithms.bootstrap(control_similarities, n_boot=1000, func=np.mean), which=95, axis=0)
        ax.fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
        ax.axhline(y=control_similarities.mean(), color='k', linestyle='-', alpha=0.1, lw=3)
    g.add_legend()
    g.set_axis_labels('silencing strength', 'response')
    g.set_titles('{col_name}')
    g.tight_layout()

    fig = g.fig

    # save figure
    fig_name = f'all_networks_all_units_fclayer_scores_vs_silencing_strength.png'
    fig.savefig(join(figures_dir, fig_name), dpi=300)
    fig_name = f'all_networks_all_units_fclayer_scores_vs_silencing_strength.pdf'
    fig.savefig(join(figures_dir, fig_name), dpi=300)

    plt.show()


with plt.rc_context(rcParams):
    # facet grid plot
    g = sns.FacetGrid(all_net_mean_scores, col='net_str', hue='type', col_wrap=4, height=2.5, aspect=3/4,
                      hue_order=[ 'inh', 'exc', 'none'],
                      palette=[ 'tab:orange', 'tab:green', 'tab:blue'])
    g.map(sns.lineplot, 'strength', 'scores', alpha=0.5, lw=3)
    # g.map(sns.scatterplot, 'strength', 'scores', alpha=0.5, s=50)
    # map also the confidence interval for the control type for each network
    for ax, net in zip(g.axes.flatten(), list_net_str):
        control_similarities = all_net_mean_scores[(all_net_mean_scores.type == 'none') & (all_net_mean_scores.net_str == net)].scores.values
        ci = sns.utils.ci(sns.algorithms.bootstrap(control_similarities, n_boot=1000, func=np.mean), which=95, axis=0)
        ax.fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
        ax.axhline(y=control_similarities.mean(), color='k', linestyle='-', alpha=0.1, lw=3)
    g.add_legend()
    g.set_axis_labels('silencing strength', 'response')
    g.set_titles('{col_name}')
    g.tight_layout()

    fig = g.fig

    # save figure
    fig_name = f'all_networks_all_units_fclayer_scores_vs_silencing_strength_no_abs.png'
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)
    fig_name = f'all_networks_all_units_fclayer_scores_vs_silencing_strength_no_abs.pdf'
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)

    plt.show()
#%%




#%%
network_list = ['alexnet',
                'resnet50', 'resnet50_linf0.5', 'resnet50_linf1', 'resnet50_linf2', 'resnet50_linf4', 'resnet50_linf8']

similarity_metric = 'cosine_similarity'
# similarity_metric = 'euclidean_norm'

def load_rsa_df_from_net_str(net_str, rootdir, preprocessed_data_dir_path, network_list, similarity_metric='cosine_similarity'):
    # Save the DataFrame if preprocessing to file is enabled
    preprocess_data_to_file = True
    df_dict = {}
    image_dict = {}
    complete_unit_index_list, layer_str, perturbation_pattern = get_complete_unit_list(net_str, rootdir)
    df_rsa = preprocess_and_compute_rsa_df(df_dict, image_dict, network_list, complete_unit_index_list,
                                           net_str, layer_str, preprocess_data_to_file, preprocessed_data_dir_path,
                                           similarity_metric=similarity_metric)

    # recover a dataframe with the mean cosine similarity for each type and strength in a tabular format
    df_rsa_table = df_rsa.stack().stack().reset_index(name=f'mean_{similarity_metric}')
    # average mean_cosine_similarity over the test_network level when grouping by type, strength and unit
    df_rsa_table = df_rsa_table.groupby(['type', 'strength', 'unit'])[f'mean_{similarity_metric}'].mean().reset_index()

    # drop units in units_to_exclude
    # df_rsa_table = df_rsa_table[~df_rsa_table.unit.isin(units_to_exclude)]

    # divide the mean_cosine_similarity by the control mean_cosine_similarity
    for group_id, group_data in df_rsa_table.groupby(['unit']):
        control_mean_similarity = group_data[group_data.type == 'none'][f'mean_{similarity_metric}'].values.mean()
        df_rsa_table.loc[df_rsa_table.unit == group_id, f'norm_mean_{similarity_metric}'] = \
            df_rsa_table.loc[df_rsa_table.unit == group_id, f'mean_{similarity_metric}'] / control_mean_similarity

    return df_rsa_table

all_net_rsa_df = pd.DataFrame()
for net_str in list_net_str:
    df_rsa_table = load_rsa_df_from_net_str(net_str, rootdir, preprocessed_data_dir_path, network_list, similarity_metric=similarity_metric)
    df_rsa_table['net_str'] = net_str

    abs_df = df_rsa_table[df_rsa_table['type'] == 'none'].replace('none', 'abs')
    inh_df = df_rsa_table[df_rsa_table['type'] == 'none'].replace('none', 'inh')
    exc_df = df_rsa_table[df_rsa_table['type'] == 'none'].replace('none', 'exc')

    # Concatenate the filtered DataFrames
    df_rsa_table = pd.concat([df_rsa_table, abs_df, inh_df, exc_df], ignore_index=True)

    if all_net_rsa_df.empty:
        all_net_rsa_df = df_rsa_table
    else:
        all_net_rsa_df = pd.concat([all_net_rsa_df, df_rsa_table], axis=0)

#%%

rcParams = {'font.size': 16,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'font.family': 'arial',
            }

with plt.rc_context(rcParams):
    g = sns.FacetGrid(all_net_rsa_df, col='net_str', hue='type', col_wrap=4, height=2.5, aspect=3/4,
                      hue_order=['abs', 'inh', 'exc', 'none'],
                      palette=['tab:purple', 'tab:orange', 'tab:green', 'tab:blue'])
    g.map(sns.lineplot, 'strength', f'mean_{similarity_metric}', alpha=0.5, lw=3)
    # g.map(sns.scatterplot, 'strength', 'scores', alpha=0.5, s=50)
    # map also the confidence interval for the control type for each network
    for ax, net in zip(g.axes.flatten(), list_net_str):
        control_similarities = all_net_rsa_df[(all_net_rsa_df.type == 'none') & (all_net_rsa_df.net_str == net)][f'mean_{similarity_metric}'].values
        ci = sns.utils.ci(sns.algorithms.bootstrap(control_similarities, n_boot=1000, func=np.mean), which=95, axis=0)
        ax.fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
        ax.axhline(y=control_similarities.mean(), color='k', linestyle='-', alpha=0.1, lw=3)
    g.add_legend()
    # g.set_axis_labels('silencing strength', 'response')
    g.set_titles('{col_name}')
    g.tight_layout()

    fig = g.fig
    # save figure
    fig_name = f'all_networks_all_units_fclayer_{similarity_metric}_vs_silencing_strength.png'
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)
    fig_name = f'all_networks_all_units_fclayer_{similarity_metric}_vs_silencing_strength.pdf'
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)

    plt.show()

with plt.rc_context(rcParams):
    g = sns.FacetGrid(all_net_rsa_df, col='net_str', hue='type', col_wrap=4, height=2.5, aspect=3 / 4,
                      hue_order=['inh', 'exc', 'none'],
                      palette=['tab:orange', 'tab:green', 'tab:blue'])
    g.map(sns.lineplot, 'strength', f'mean_{similarity_metric}', alpha=0.5, lw=3)
    # g.map(sns.scatterplot, 'strength', 'scores', alpha=0.5, s=50)
    # map also the confidence interval for the control type for each network
    for ax, net in zip(g.axes.flatten(), list_net_str):
        control_similarities = all_net_rsa_df[(all_net_rsa_df.type == 'none') & (all_net_rsa_df.net_str == net)][
            f'mean_{similarity_metric}'].values
        ci = sns.utils.ci(sns.algorithms.bootstrap(control_similarities, n_boot=1000, func=np.mean), which=95, axis=0)
        ax.fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
        ax.axhline(y=control_similarities.mean(), color='k', linestyle='-', alpha=0.1, lw=3)
    g.add_legend()
    # g.set_axis_labels('silencing strength', 'response')
    g.set_titles('{col_name}')
    g.tight_layout()

    fig = g.fig
    # save figure
    fig_name = f'all_networks_all_units_fclayer_{similarity_metric}_vs_silencing_strength_no_abs.png'
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)
    fig_name = f'all_networks_all_units_fclayer_{similarity_metric}_vs_silencing_strength_no_abs.pdf'
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)

    plt.show()

#%%
####################### Section to compute and plot the robustness of the networks ############################
# extract only net_str containing resnet50
resnet50_df = all_net_rsa_df[all_net_rsa_df.net_str.str.contains('resnet50')]
# cretae robustness column by regex of the net_str column resnet50_linf\d include floatin point number, as optional
resnet50_df['robustness'] = resnet50_df.net_str.str.extract(r'resnet50_linf(\d+\.?\d*)').astype(float)
# replace nan robustness with 0
resnet50_df['robustness'] = resnet50_df['robustness'].fillna(0)
resnet50_df = resnet50_df.reset_index(drop=True)
#%%
# Function to calculate delta_mean_cosine_similarity
def calculate_delta_mean_cosine_similarity(df):
    # Get the mean_cosine_similarity where type is 'none'
    mean_none = df.loc[df['strength'] == 0, 'mean_cosine_similarity'].mean()
    # Subtract this value from all mean_cosine_similarity values in the group
    return df['mean_cosine_similarity'] - mean_none

# Apply the function to each group
delta_series = resnet50_df.groupby(['type', 'unit', 'net_str']).apply(calculate_delta_mean_cosine_similarity).reset_index(level=[0,1,2 ], drop=True)
# Add the delta_mean_cosine_similarity column to the original DataFrame
resnet50_df.loc[delta_series.index, 'delta_mean_cosine_similarity'] = delta_series
#%%
from scipy.stats import pearsonr, spearmanr
# compute correlation between robustness and delta_mean_cosine_similarity with pvlaue, per type
# correlation, pvalue = spearmanr(resnet50_df.robustness, resnet50_df.delta_mean_cosine_similarity)

group_correlation = resnet50_df.groupby(['type', 'strength']).apply(lambda x: spearmanr(x.robustness, x.delta_mean_cosine_similarity))
# make a dataframe from the groupby object
group_correlation_df = pd.DataFrame(group_correlation.to_list(), index=group_correlation.index, columns=['correlation', 'pvalue'])
group_correlation_df.reset_index(inplace=True)

table_correlation = group_correlation_df.copy()
style_format = {'correlation': '{:.2f}', 'pvalue': '{:.0e}', 'strength': '{:.1f}'}

for key, value in style_format.items():
    table_correlation[key] = table_correlation[key].apply(lambda x: value.format(x))


# multicolumn, first level type, second level strength, correlation and pvalue
table_correlation = table_correlation[table_correlation.type.isin(['exc', 'inh'])]
table_correlation = table_correlation.pivot(index='strength', columns='type').swaplevel(axis=1).sort_index(axis=1)
# merge correlation and pvalue columns as correlation (pvalue) in a new column
# loop over first level columns
for col in table_correlation.columns.levels[0]:
    table_correlation_tmp = table_correlation[col, 'correlation'] + ' (' + table_correlation[col, 'pvalue'] + ')'
    # new column name
    new_col = (col, 'correlation (pvalue)')
    table_correlation[new_col] = table_correlation_tmp
    table_correlation.drop(columns=[(col, 'correlation'), (col, 'pvalue')], inplace=True)


#%%





# print the table
print(table_correlation.to_latex())


#%%
# plot group_correlation_df correlation vs strength, with hue per type
with plt.rc_context(rcParams):
    sns.lineplot(data=group_correlation_df, x='strength', y='correlation', hue='type')
    plt.show()
#%%
# plot resnet50_df delta_mean_cosine_similarity vs robustness
with plt.rc_context(rcParams):
    sns.lineplot(data=resnet50_df[resnet50_df.strength==1], x='robustness', y='delta_mean_cosine_similarity', hue='strength', style='type', markers=True)
    # sns.barplot(data=resnet50_df[resnet50_df.strength==1], x='robustness', y='delta_mean_cosine_similarity', hue='strength', ci=95)
    sns.scatterplot(data=resnet50_df[resnet50_df.strength==1], x='robustness', y='delta_mean_cosine_similarity', hue='type')
    # log x scale
    plt.xscale('symlog')
    # write original xticks, from robustness column
    plt.xticks(ticks=resnet50_df.robustness.unique(), labels=resnet50_df.robustness.unique())
    # ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('robustness')
    plt.ylabel('$\Delta$(cosine similarity)')
    plt.title('Ablation effect vs robustness in ResNet50')
    plt.show()

#%%
# plot resnet50_df delta_mean_cosine_similarity vs robustness with column per type, hue per strength
with plt.rc_context(rcParams):
    g = sns.FacetGrid(resnet50_df[resnet50_df.type.isin(['exc', 'inh'])], col='type', hue='strength',
                      col_wrap=2, height=3, aspect=1, palette='viridis')
    g.map(sns.lineplot, 'robustness', 'delta_mean_cosine_similarity')
    plt.xscale('symlog')
    plt.xticks(ticks=resnet50_df.robustness.unique(), labels=resnet50_df.robustness.unique())
    # save figure
    fig_name = f'resnet50_delta_mean_cosine_similarity_vs_robustness_per_type_per_strength.png'
    fig = g.fig
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)
    fig_name = f'resnet50_delta_mean_cosine_similarity_vs_robustness_per_type_per_strength.pdf'
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)
    plt.show()

    g = sns.FacetGrid(resnet50_df[resnet50_df.type.isin(['exc', 'inh'])], col='type', hue='strength',
                      col_wrap=2, height=3, aspect=1, palette='viridis')
    g.map(sns.lineplot, 'robustness', 'mean_cosine_similarity')
    g.add_legend(title='silencing strength')
    plt.xscale('symlog')
    plt.xticks(ticks=resnet50_df.robustness.unique(), labels=resnet50_df.robustness.unique())
    # save figure
    fig_name = f'resnet50_mean_cosine_similarity_vs_robustness_per_type_per_strength.png'
    fig = g.fig
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)
    fig_name = f'resnet50_mean_cosine_similarity_vs_robustness_per_type_per_strength.pdf'
    fig.savefig(join(figures_dir, fig_name), dpi=300, transparent=True)

    plt.show()
#%%

eeeeeee

#%%

# section header in comments
#====================================================================================================
# Plotting images across experiments
#====================================================================================================


for unit, df_temp in df_dict.items():
    print(unit)
    df_temp = df_temp.copy()
    # delete rows with df_temp['strength'] ~= 0 or 1
    df_temp = df_temp[df_temp['strength'].isin([0, 1])]

    # Explode score column
    score_col = [col for col in df_temp.columns if 'scores' in col][0]
    df_temp = df_temp.explode(score_col)
    df_temp.rename(columns={score_col: 'scores'}, inplace=True)
    df_temp.reset_index(inplace=True)
    df_temp.rename(columns={'index': 'old_index'}, inplace=True)

    # Calculate necessary values
    n_types = len(df_temp['type'].unique())
    n_top = df_temp.groupby(['type', 'strength']).size().max()
    max_n_ims_type = df_temp.groupby(['type', 'strength']).size().reset_index().groupby('type')[
        'strength'].nunique().max()

    # with mpl.rc_context(params):
    #     # Plot all images
    #     all_ims_grid = make_grid(image_dict[unit][1], nrow=2)
    #     plt.imshow(all_ims_grid.permute(1, 2, 0))
    #     plt.axis('off')
    #     plt.title(f'{net_str} {layer_str} unit {unit}')
    #     plt.show()

    # Get top images for each type
    image_list = image_dict[unit]
    top_n = 4
    tmp_grid_list = []

    for name, group in df_temp.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength']).head(
            1).groupby('type'):
        for _, row_data in group.iterrows():
            list_index = row_data.old_index
            top_inds = anevo.list_argsort(df_temp[df_temp.old_index == list_index].scores.tolist(), reverse=True)[
                       :top_n]
            images = image_list[list_index][top_inds, ...]
            tmp_grid = make_grid(images, nrow=2, padding=5)
            tmp_grid_list.append(tmp_grid)


    # Add padding to tmp_grid_list, start with control image and continue on in intervals of n_types
    padded_tmp_grid_list = []
    for i in range(n_types - 1):
        padded_tmp_grid_list += [tmp_grid_list[0]] + tmp_grid_list[i * max_n_ims_type: (i + 1) * max_n_ims_type]


    group_keys = list(df_temp.groupby(['type', 'strength']).groups.keys())

    # Plot grid of images
    # map index list to subplot index list in intervals of max_n_ims_type, appending the control image at the beginning
    indices_to_subplot = []
    for i in range(n_types - 1):
        indices_to_subplot +=  [-1] + list(range(i * max_n_ims_type, (i + 1) * max_n_ims_type))
    fig, axes = plt.subplots(nrows=n_types - 1, ncols=max_n_ims_type + 1, figsize=(1.5 * (max_n_ims_type + 1), 1.5 * (n_types - 1)), dpi=300)
    for ii, tmp_grid in enumerate(padded_tmp_grid_list):
        plt.sca(axes[ii // (max_n_ims_type + 1), ii % (max_n_ims_type + 1)])
        plt.imshow(tmp_grid_list[indices_to_subplot[ii]].permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'{group_keys[indices_to_subplot[ii]][0]} {group_keys[indices_to_subplot[ii]][1]}')
    plt.tight_layout()
    plt.show()
    if compile_all_units:
        fig_filename = os.path.join(figures_dir, f'{net_str}_{layer_str}_unit_{unit}_top{top_n}_images.pdf')
    break

#%%
# add the last list to positions 0, ntypes, 2*ntypes
padded_tmp_grid_list = [tmp_grid_list[0]] + tmp_grid_list[:n_types] + \
                [tmp_grid_list[0]] + tmp_grid_list[n_types: 2*n_types] +\
                [tmp_grid_list[0]] + tmp_grid_list[2*n_types:-1]
# make a list of group names after type and strength
group_keys = list(df_temp.groupby(['type', 'strength']).groups.keys())
fig, ax = plt.subplots(figsize=(15, 9), dpi=300)
# make a grid of the first 12 images from tmp_grid_list
all_ims_grid = make_grid(padded_tmp_grid_list, nrow=5, padding=30, pad_value=1)
plt.imshow(all_ims_grid.permute(1, 2, 0))
plt.axis('off')
plt.title('{labels[ii][0]}')
plt.show()

#%%
ssdxx
sys.exit()

#%%
# y_var = 'norm_mean_cosine_similarity'  #
# y_var = 'mean_cosine_similarity'


#
# #plot a horizontal line at the control cosine similarity df[df.type == 'none'].cosine_similarity.values.mean()
# control_similarities = df_rsa_table[df_rsa_table.type == 'none'][y_var].values
# # then compute the confidence interval
# ci = sns.utils.ci(sns.algorithms.bootstrap(control_similarities, n_boot=1000, func=np.mean), which=95, axis=0)
# # plot the mean and confidence interval, without edgecolor
# plt.fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
# plt.axhline(y=control_similarities.mean(), color='k', linestyle='-', alpha=0.1)
# # plot the cosine similarity vs the silencing strength separately for each type, using seaborn
# sns.lineplot(data=df_rsa_table, x='strength', y=y_var, hue='type', alpha=0.5)
#
# plt.title(f'{net_str} {layer_str}')
# # square axes, with unequal aspect ratio
# plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
# # move legend outside the plot
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.tight_layout()
# plt.show()


# # make three subplots, one for each type in a row
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 3), dpi=300, sharex=True, sharey=True)
#
# for ii, type in enumerate(['inh', 'exc', 'abs']):
#     axes[ii].set_title(type)
#     # plot the mean and confidence interval of the control df_rsa[df_rsa.type == 'none'].mean_cosine_similarity
#     control_similarities = df_rsa_table[df_rsa_table.type == 'none'][y_var].values
#     # first compute the mean
#     mean = control_similarities.mean()
#     # then compute the confidence interval
#     ci = sns.utils.ci(sns.algorithms.bootstrap(control_similarities, n_boot=1000, func=np.mean), which=95, axis=0)
#     # plot the mean and confidence interval, without edgecolor
#     axes[ii].fill_between(x=[0, 1], y1=ci[0], y2=ci[1], color=[0.9, 0.9, 0.9], alpha=0.5, edgecolor=None)
#     axes[ii].axhline(y=mean, color='k', linestyle='-', alpha=0.1)
#     # plot the scatter values of the control df_rsa_table[df_rsa_table.type == 'none'][y_var]
#     sns.scatterplot(data=df_rsa_table[df_rsa_table.type == 'none'], x='strength', y=y_var, hue='unit',
#                     ax=axes[ii], palette='tab20', alpha=1)
#     sns.lineplot(data=df_rsa_table[df_rsa_table.type == type], x='strength', y=y_var, hue='unit',
#                  ax=axes[ii], palette='tab20', alpha=1)
#     # axes[ii].axhline(y=df_rsa_table[df_rsa_table.type == 'none'][y_var].values.mean(), color='k', linestyle='--')
#     # axes[ii].set_xlim([0, 1])
#     # axes[ii].set_ylim([0, 1])
#     axes[ii].set_xticks([0, 0.5, 1])
#     axes[ii].set_ylabel('cosine similarity(silenced, control)')
#     axes[ii].set_xlabel('silencing strength')
#     axes[ii].legend().remove()
#
# # fig title using net_str and layer_str
# fig.suptitle(f'{net_str} {layer_str}')
# fig.tight_layout()
# plt.show()
    #%%
if 0:
    # for each row in df, get the image list from image_list corresponding to the list_index
    # and then compute the output from several networks
    all_ims_tensor = torch.stack(top_per_condition_im_list)

    network_list = ['alexnet',
                    'resnet50',
                    'resnet50_linf0.5', 'resnet50_linf1', 'resnet50_linf2', 'resnet50_linf4', 'resnet50_linf8']
    model_outputs = []
    # Now pass the tensor to get network activations

    for net in network_list:
        scorer = TorchScorer(net)
        # for each row in df, get the image list from image_list corresponding to the list_index
        # and then compute the output from several networks
        # find the index of df where type is none

        control_im_list = image_list[np.where(df.type == 'none')[0][0]]  # here one image is from one evolution
        control_tensor = torch.stack(control_im_list).to(torch.device('cuda:0'))

        for index, row in df.iterrows():
            # get the image list from image_list corresponding to the list_index
            silenced_images_tensor = torch.stack(image_list[index]).to(torch.device('cuda:0'))
            control_output = scorer.model(control_tensor)
            silenced_output = scorer.model(silenced_images_tensor)
            # compute softmax
            control_output = torch.nn.functional.softmax(control_output, dim=1)
            silenced_output = torch.nn.functional.softmax(silenced_output, dim=1)
            # compute the correlation matrix between control and silenced
# stopped here, continue with cosine of torchmetrics

        model_output = scorer.model(all_ims_tensor.to(torch.device('cuda:0'))).cpu().detach()
        model_output = torch.nn.functional.softmax(model_output, dim=1)
        # we know we picked topn images per group
        if topn > 1:
            model_output = model_output.reshape(-1, topn, 1000).mean(axis=1)
        model_outputs.append(
            torch.log(model_output))  # cross entropy loss uses sum over one-hot class x log(softmax(class))

    # this sorting could be done by list_argsort list by list instead of exploding the dataframe
    # Get the list of top n images sorted in same order as dataframe
    topn = 1
    topn_df = df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength']).apply(
        lambda group: group.sort_values(unit_scores_str, ascending=False).head(topn))

    # Select only one index of multiindex DataFrame
    # https://stackoverflow.com/questions/28140771/select-only-one-index-of-multiindex-dataframe
    top_indices = topn_df.index.get_level_values(2).values

    # top_per_condition_im_list = [im
    #                              for ind, im in
    #                              enumerate(itertools.chain.from_iterable(perturbation_data_dict['images']))
    #                              if ind in top_indices]
    # top_per_condition_im_list = [top_per_condition_im_list[sorted(top_indices).index(ind)] for ind in top_indices]

#%%
# do not try to merge an exploded list, there are duplicates in the columns type and strength resulting in memory error
df_list = list(df_dict.values())

merged_df = reduce(lambda left, right:
                   pd.merge(left, right, on=['type', 'strength'], how='inner', validate='one_to_one'),
                   df_list)

#%%

test_df = pd.merge(df_list[0], df_list[1], how='inner', on=['type', 'strength'])

#%%
score_name = 'scores_217'
df = df.explode([score_name])
# df = df.explode(['scores'])

df.scores = df[score_name].astype("float")
df.type = df.type.astype("string")
df = df.reset_index(names='old_index')
max_inds = df.sort_values(score_name, ascending=False).drop_duplicates(['type', 'strength']).index.values

#%%
# Get maximum image per condition
max_df = df.sort_values(['type', 'strength'], ascending=True).groupby(['type', 'strength'], group_keys=False).apply(
    lambda group: group.sort_values('scores', ascending=False).head(1))  # set group_keys=False to avoid multiindex creation

#%%
max_per_condition_im_list = [im for ind, im in enumerate(itertools.chain.from_iterable(perturbation_data_dict['images']))
                             if ind in max_inds]
# Sort so the order matches the max_df
max_per_condition_im_list = [max_per_condition_im_list[sorted(max_inds).index(ind)] for ind in max_df.index]
#%% # start of original code ##########################################################################################
# layer_str = '.classifier.Linear6'
# layer_str = '.Linearfc' # for resnets
unit_idx = 574#398# imagenet 373  # ecoset 13, 373, 14, 12, 72, 66, 78
unit_pattern = 'alexnet-eco-080.*%s_%s' % (layer_str, unit_idx)
# unit_pattern = 'alexnet_.*%s_%s' % (layer_str, unit_idx)
unit_pattern = 'resnet50_%s_%s' % (layer_str, unit_idx)
unit_pattern = 'resnet50_linf0.5_%s_%s' % (layer_str, unit_idx)
unit_pattern = 'resnet50_linf8_%s_%s' % (layer_str, unit_idx)

unit_pattern = net_str + '_' + layer_str + '_' + str(unit_idx)

perturbation_pattern = '_kill_topFraction_'
# %%
original_dir = [idir for idir in next(os.walk(rootdir))[1] if re.match(unit_pattern + '$', idir)]
data_dirs = [idir for idir in next(os.walk(rootdir))[1] if re.match(unit_pattern + perturbation_pattern, idir)]

#%%
# get a list from the perturbation suffixes _kill.*$
suffix_pattern = re.compile(r'.*(_kill.*)$')
perturbation_suffixes = [re.search(suffix_pattern, dir).group(1) for dir in data_dirs]


# %%

columns = ['type', 'strength', 'scores']  # ['type', 'strength', 'scores', 'images'] don't put images! Too big
df = pd.DataFrame.from_dict({key: perturbation_data_dict[key] for key in columns})
df = df.explode(['scores'])
# Put proper types to avoid object type
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
max_per_condition_im_list = [im
                             for ind, im in enumerate(itertools.chain.from_iterable(perturbation_data_dict['images']))
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
figures_dir = os.path.join('C:', 'Users', 'gio', 'Data', 'figures', 'silencing')
os.makedirs(figures_dir, exist_ok=True)
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
    plt.savefig(os.path.join(figures_dir, '%s__net_output_correlations.pdf' % original_dir[0]), dpi=300, format='pdf', metadata=None,
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

