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
from PIL import Image


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
    print(resized_tensor.shape)
    with torch.no_grad():
        output_resized = model(resized_tensor.to('cuda'))
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

dataset_name = 'imagenet-o' #https://github.com/hendrycks/natural-adv-examples/blob/master/README.md
dataset_path = r'N:\PonceLab\Users\Giordano\Data\imagenet-o'


#%%

# define this "M:\Documents\Training-data-metadata\imagenet_classes.txt" in an os agnostic way
imagenet_metadata_path = join('M:\\', 'Documents', 'Training-data-metadata', 'imagenet_classes.txt')
with open(imagenet_metadata_path, 'r') as f:
    imagenet_index2class = f.readlines()

imagenet_index2class = [line.strip() for line in imagenet_index2class]

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

# List to store paths of all image files
image_file_paths = []

# Walk through directory and its subfolders
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_file_paths.append(os.path.join(root, file))


# Output the list of image files
for image in image_file_paths:
    print(image)

#%%
dirs
# make a dataloader with the image_paths,
# then iterate over the dataloader to get the images
# then pass the images to the object detection function
# then save the objectness metrics to a dataframe
# then save the dataframe to a pickle file

# make dataset from file paths
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None, filepaths=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_paths = self._get_image_filepaths(filepaths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.folder_path, self.image_files[idx])
        img_name = self.file_paths[idx]
        image = Image.open(img_name).convert('RGB')  # Ensure image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image

    def _get_image_filepaths(self, filepaths):
        if filepaths is not None:
            return filepaths
        else:
            valid_extensions = ('.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp')
            image_files = [file for file in os.listdir(self.folder_path) if file.lower().endswith(valid_extensions)]
            return [os.path.join(self.folder_path, file) for file in image_files]
#
# dataset = ImageFolderDataset(dataset_path, transform=ToTensor(), filepaths=image_file_paths)
# dataloader = DataLoader(dataset, batch_size=100, shuffle=False)





#%%
compute_objectness = False

model.to('cuda')

if compute_objectness:
    # get objectness for each list of images in the image_files
    # Initialize dictionary to store objectness metrics
    objectness_dict = {
        'mean_objectness': [],
        'max_objectness': [],
        'n_objects': []
    }

    # Process image_file_paths
    for image_file_path in tqdm(image_file_paths):
        # Load image
        condition_img = Image.open(image_file_path).convert('RGB')
        condition_img_tensor = ToTensor()(condition_img).unsqueeze(0)

        try:
            # Compute objectness metrics
            tmp_objectness, tmp_max_objectness, tmp_n_objects = get_image_objectness(condition_img_tensor, model, stride=32)

            # Append results to the lists
            objectness_dict['mean_objectness'].append(tmp_objectness)
            objectness_dict['max_objectness'].append(tmp_max_objectness)
            objectness_dict['n_objects'].append(tmp_n_objects)
        except Exception as e:
            print(f"Error processing image {image_file_path}: {e}")
            continue

    # Convert the objectness_dict to a DataFrame
    objectness_df = pd.DataFrame(objectness_dict)

    # If you have other DataFrames to merge with, use a similar approach
    # df_score_objectness = pd.merge(other_df, objectness_df, left_index=True, right_index=True)

    # Save the objectness DataFrame to a pickle file
    file_name = f'{dataset_name}_dataframe_objectness.pickle'
    with open(join(preprocessed_data_dir_path, file_name), 'wb') as handle:
        pickle.dump(objectness_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
objectness_df = pd.read_pickle(join(preprocessed_data_dir_path, f'{dataset_name}_dataframe_objectness.pickle'))


# list_bool_df = (objectness_df.map(type) == list).all()
# list_columns = list_bool_df[list_bool_df].index.tolist()
# objectness_df = objectness_df.explode(list_columns)
#%%
objectness_df = objectness_df.explode(objectness_df.columns.values.tolist())
#%%
# use default pycharm mpl backend
# print backends available
print(mpl.rcsetup.all_backends)
# set rhe best for pycharm
# mpl.use('Qt5Agg')
mpl.use('module://backend_interagg')  # or 'module://backend_agg' for non-interactive
#%%

import seaborn as sns

sns.histplot(objectness_df['max_objectness'])
plt.show()

sns.histplot(objectness_df['mean_objectness'])
plt.show()

#%%
from scipy import stats
# mean and std of max objectness
mean_max_objectness = objectness_df['max_objectness'].mean()
std_max_objectness = objectness_df['max_objectness'].std()

print(f'Max objectness (mean +/- std): {mean_max_objectness:.2f} +/- {std_max_objectness:.2f}')

ci_max_objecteness = stats.bootstrap(
    (objectness_df['max_objectness'].values,), np.mean, confidence_level=0.95, n_resamples=1000)

#print mean and ci in parenthesis with 0.2f precision
print('mean = %.2f (%.2f, %.2f)' % (mean_max_objectness,
                                    ci_max_objecteness.confidence_interval.low,
                                    ci_max_objecteness.confidence_interval.high))

# plot the histogram of bootstrapped means
sns.histplot(ci_max_objecteness.bootstrap_distribution, bins=30)
plt.show()
#%%
# sample biggan images

import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

# Assuming you have BigGAN pre-trained model from torchvision
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names, save_as_images) # install it from huggingface GR
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN

# Load the pre-trained BigGAN model
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

class BigGAN_wrapper(): #nn.Module
    def __init__(self, BigGAN, ):
        self.BigGAN = BigGAN

    def sample_vector(self, sampn=1, class_id=None, device="cuda", noise_std=0.7):
        if class_id is None:
            refvec = torch.cat((noise_std * torch.randn(128, sampn).to(device),
                                self.BigGAN.embeddings.weight[:, torch.randint(1000, size=(sampn,))].to(device),)).T
        else:
            refvec = torch.cat((noise_std * torch.randn(128, sampn).to(device),
                                self.BigGAN.embeddings.weight[:, (class_id*torch.ones(sampn)).long()].to(device),)).T
        return refvec

    def visualize(self, code, scale=1.0, truncation=0.7):
        imgs = self.BigGAN.generator(code, truncation)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, truncation=0.7, B=15, ):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        with torch.no_grad():
            while csr < imgn:
                csr_end = min(csr + B, imgn)
                code_batch = torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda()
                img_list = self.visualize(code_batch, truncation=truncation, ).cpu()
                img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
                csr = csr_end
        return img_all

    def render(self, codes_all_arr, truncation=0.7, B=15):
        img_tsr = self.visualize_batch_np(codes_all_arr, truncation=truncation, B=B)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]


#%%
# make a folder to store biggan
biggan_dir = join(rootdir, 'BigGAN_images_rebuttal')
if not os.path.exists(biggan_dir):
    os.makedirs(biggan_dir)

# fc6 dir
fc6_dir = join(rootdir, 'fc6_images_rebuttal')
if not os.path.exists(fc6_dir):
    os.makedirs(fc6_dir)



model_gan = load_GAN("BigGAN")

# sample 5 batches of 200 images
n_images = 1000
batch_size = 200
n_batches = n_images // batch_size

# set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

if 0:
    for i in tqdm(range(n_batches)):
            code = model_gan.sample_vector(batch_size).cuda()
            samples = model_gan.visualize(code)
            save_image(samples, join(biggan_dir, f'biggan_images_{i}.png'), nrow=10)

    #%%
if 0:
    model_gan = load_GAN("fc6")
    # sample 5 batches of 200 images

    for i in tqdm(range(n_batches)):
        code = model_gan.sample_vector(batch_size).cuda()
        samples = model_gan.visualize(code)
        save_image(samples, join(fc6_dir, f'fc6_images_{i}.png'), nrow=10)


#%%
# undo the make grid operation
import math


def reverse_make_grid(grid, nrows, image_size, padding=0):
    """
    Reverses the operation of torchvision.utils.make_grid.

    Args:
        grid (torch.Tensor): The grid tensor (e.g., from make_grid) of shape (C, H, W).
        nrows (int): The number of rows used to create the grid.
        image_size (tuple): The size of each image (height, width) as (px, py).
        padding (int): The padding applied between images.

    Returns:
        torch.Tensor: A tensor of shape (N, C, px, py), where N is the number of images.
    """
    C, H, W = grid.size()
    px, py = image_size

    # Calculate the number of images (assuming grid is rectangular)
    ncols = (H + padding) // (py + padding)
    print(nrows, ncols)
    n_images = nrows * ncols

    # Create a list to hold the extracted images
    images = []

    for i in range(ncols):
        for j in range(nrows):
            # Calculate the position of each image in the grid
            y1 = i * (px + padding)
            y2 = y1 + px
            x1 = j * (py + padding)
            x2 = x1 + py

            # Extract the image slice and append it to the list
            image = grid[:, y1:y2, x1:x2]
            images.append(image)

    # Stack the images into a batch
    images = torch.stack(images)

    return images


#%%

# process biggan images with yolo
# Initialize dictionary to store objectness metrics
objectness_dict_biggan = {'mean_objectness': [], 'max_objectness': [], 'n_objects': []}

# Process image_file_paths
for i in tqdm(range(n_batches)):
    # Load image
    condition_img = Image.open(join(biggan_dir, f'biggan_images_{i}.png')).convert('RGB')
    # split image grid into individual images, opposite of make_grid
    condition_img_tensor = reverse_make_grid(ToTensor()(condition_img), 10, (256,256), padding=2)
    # try:
        # Compute objectness metrics
        tmp_objectness, tmp_max_objectness, tmp_n_objects = get_image_objectness(condition_img_tensor, model, stride=32)

        # Append results to the lists
        objectness_dict_biggan['mean_objectness'].append(tmp_objectness)
        objectness_dict_biggan['max_objectness'].append(tmp_max_objectness)
        objectness_dict_biggan['n_objects'].append(tmp_n_objects)
    # except Exception as e:
    #     print(f"Error processing image {i}: {e}")
    #     continue

objectness_df_biggan = pd.DataFrame(objectness_dict_biggan)


#%%
# Save the objectness DataFrame to a pickle file
file_name = f'biggan_dataframe_objectness.pickle'
with open(join(preprocessed_data_dir_path, file_name), 'wb') as handle:
    pickle.dump(objectness_df_biggan, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%

# check which columns are lists
objectness_df_biggan = objectness_df_biggan.explode(objectness_df_biggan.columns.values.tolist())
# mean and std of max objectness
mean_max_objectness = objectness_df_biggan['max_objectness'].mean()
std_max_objectness = objectness_df_biggan['max_objectness'].std()

print(f'Max objectness (mean +/- std): {mean_max_objectness:.2f} +/- {std_max_objectness:.2f}')


#%%
# now do the fc6 processing with yolo

# Initialize dictionary to store objectness metrics
objectness_dict_fc6 = {'mean_objectness': [], 'max_objectness': [], 'n_objects': []}

for i in tqdm(range(n_batches)):
    # Load image
    condition_img = Image.open(join(fc6_dir, f'fc6_images_{i}.png')).convert('RGB')
    # split image grid into individual images, opposite of make_grid
    condition_img_tensor = reverse_make_grid(ToTensor()(condition_img), 10, (256,256), padding=2)
    # try:
        # Compute objectness metrics
    tmp_objectness, tmp_max_objectness, tmp_n_objects = get_image_objectness(condition_img_tensor, model, stride=32)

    # Append results to the lists
    objectness_dict_fc6['mean_objectness'].append(tmp_objectness)
    objectness_dict_fc6['max_objectness'].append(tmp_max_objectness)
    objectness_dict_fc6['n_objects'].append(tmp_n_objects)
    # except Exception as e:
    #     print(f"Error processing image {i}: {e}")
    #     continue

objectness_df_fc6 = pd.DataFrame(objectness_dict_fc6)

# Save the objectness DataFrame to a pickle file
file_name = f'fc6_dataframe_objectness.pickle'
with open(join(preprocessed_data_dir_path, file_name), 'wb') as handle:
    pickle.dump(objectness_df_fc6, handle, protocol=pickle.HIGHEST_PROTOCOL)

objectness_df_fc6 = pd.read_pickle(join(preprocessed_data_dir_path, f'fc6_dataframe_objectness.pickle'))
objectness_df_fc6 = objectness_df_fc6.explode(objectness_df_fc6.columns.values.tolist())

#%%

# mean and std of max objectness
mean_max_objectness = objectness_df_fc6['max_objectness'].mean()
std_max_objectness = objectness_df_fc6['max_objectness'].std()

print(f'Max objectness (mean +/- std): {mean_max_objectness:.2f} +/- {std_max_objectness:.2f}')

#%%
# merge the dataframes for biggan and fc6
objectness_df_biggan['gan'] = 'biggan'
objectness_df_fc6['gan'] = 'fc6'


objectness_df = pd.concat([objectness_df_biggan, objectness_df_fc6], axis=0)
#%%
# plot distributions of max objectness for biggan and fc6
sns.histplot(objectness_df, x='max_objectness', hue='gan', kde=True)
plt.show()

# mean and std of max objectness
mean_max_objectness = objectness_df['max_objectness'].mean()
std_max_objectness = objectness_df['max_objectness'].std()

print(f'Max objectness (mean +/- std): {mean_max_objectness:.2f} +/- {std_max_objectness:.2f}')


# import libraries to do confidence interval with bootstrapping
from scipy import stats
# get 95% confidence interval by bootstrapping
ci_max_objecteness = stats.bootstrap(
    (objectness_df['max_objectness'].values,), np.mean, confidence_level=0.95, n_resamples=1000)

#print mean and ci in parenthesis with 0.2f precision
print('mean = %.2f (%.2f, %.2f)' % (mean_max_objectness,
                                    ci_max_objecteness.confidence_interval.low,
                                    ci_max_objecteness.confidence_interval.high))
#%%


ci_max_objectness_by_gan = objectness_df.groupby('gan')['max_objectness'].apply(lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x)))


#%%
# import make_grid from torchvision.utils
from torchvision.utils import make_grid, save_image

samples = model.visualize(model.sample_vector(100).cuda())

# Create a grid of images
grid = make_grid(samples, nrow=10)
plt.imshow(ToPILImage()(grid))
plt.show()

#%%
