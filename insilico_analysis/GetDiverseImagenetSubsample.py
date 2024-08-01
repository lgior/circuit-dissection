import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

# Define a custom dataset
class ImageNetDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Path to ImageNet class images (one image per class for feature extraction)
imagenet_path = r"N:\PonceLab\Stimuli\imagenet"
imagenet_train_path = os.path.join(imagenet_path, "train")
imagenet_val_path = os.path.join(imagenet_path, "val")
class_images = [os.path.join(imagenet_val_path, img) for img in os.listdir(imagenet_val_path)]

#%%
# Create dataset and dataloader
dataset = ImageFolder(imagenet_val_path, transform)
dataloader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=4)

#%%
# Load pre-trained model and move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).to(device)
model.eval()



# Function to extract features
def extract_features(dataloader, model, device):
    features = []
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)  # Move batch to the same device as the model
            output = model(batch)
            features.append(output.cpu())
    return torch.cat(features, dim=0)

# Extract features for each batch
features_tensor = extract_features(dataloader, model, device)

# Normalize features
features_tensor = normalize(features_tensor.numpy(), norm='l2', axis=1)

# Compute pairwise distances using PyTorch
dist_matrix = torch.cdist(torch.tensor(features_tensor), torch.tensor(features_tensor), p=2).numpy()

#%%
# visualizing the distance matrix
import matplotlib.pyplot as plt
plt.imshow(dist_matrix)
plt.colorbar()
plt.show()
#%%
# set all seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Perform hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=100, metric='precomputed', linkage='complete')
clustering.fit(dist_matrix)
#%%
# Select one representative class per cluster
selected_classes = []
for cluster_id in range(100):
    cluster_indices = np.where(clustering.labels_ == cluster_id)
    cluster_labels = set([dataset.samples[idx][1] for idx in cluster_indices[0]]) - set(selected_classes)
    # discard from cluster indices the ones which label is already in selected classes
    selected_classes.append(cluster_labels.pop())
print("Selected classes:", selected_classes)

#%%
# plot the first 2 pca components of the features and color by cluster
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_tensor)
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clustering.labels_)
plt.colorbar()
plt.show()
#%%
# do it in tsne
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
features_tsne = tsne.fit_transform(features_tensor)
#%%

# plot only the selected classes, find the first image of each class
selected_class_images = []
for selected_class in selected_classes:
    # find index of first sample with label == selected_class
    idx = next(idx for idx, (_, label) in enumerate(dataset.samples) if label == selected_class)
    selected_class_images.append(idx)

#%%
# plot the tsne of the selected classes
plt.scatter(features_tsne[selected_class_images, 0], features_tsne[selected_class_images, 1],
            c=clustering.labels_[selected_class_images], cmap='tab20')

plt.colorbar()
plt.show()

#%%
# plot the images of the selected classes
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
import torchvision

# plot original images without normalization
# reverse the normalization

def unnormalize_imagenet(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return image_tensor * std + mean

# collect images as a tensor
image_tensor = torch.stack([dataset[idx][0] for idx in selected_class_images])
# unnormalize the images
image_tensor = unnormalize_imagenet(image_tensor)
# make a grid with torchvision
grid = torchvision.utils.make_grid(image_tensor, nrow=10, padding=10)
# plot the grid
plt.figure(figsize=(20, 20))
plt.imshow(ToPILImage()(grid))
plt.axis('off')
plt.show()
#%%
# Save features, distance matrix, and selected classes
save_path = r"C:\Users\gio\OneDrive - Harvard University\Data\Data_ephys\PonceLab\neurips_revision\subsample_imagenet"
np.save(os.path.join(save_path, "features.npy"), features_tensor)
np.save(os.path.join(save_path, "dist_matrix.npy"), dist_matrix)
np.save(os.path.join(save_path, "selected_classes.npy"), selected_classes)
#%%