"""
Data Processing, Loading, and Augmentation Tools
"""
import src
import torch
import PIL
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_data(base_path="../data/givens/"):
    """
    Load train and test features and labels, and add file paths to image data.

    Args:
        base_path (str): The base directory where the data is stored.

    Returns:
        tuple: A tuple containing:
            - train_features (pd.DataFrame): Training feature metadata with filepaths.
            - test_features (pd.DataFrame): Test feature metadata with filepaths.
            - train_labels (pd.DataFrame): Training labels as a DataFrame.
            - species_labels (list): Sorted list of species labels (columns in train_labels).
    """
    # Load features and labels
    train_features = pd.read_csv(f"{base_path}train_features.csv", index_col="id")
    test_features = pd.read_csv(f"{base_path}test_features.csv", index_col="id")
    train_labels = pd.read_csv(f"{base_path}train_labels.csv", index_col="id")

    # Subdirectories for train and test images
    train_images_path = os.path.join(base_path, "train_features")
    test_images_path = os.path.join(base_path, "test_features")

    # Add a 'filepath' column with the full path to each image
    train_features['filepath'] = train_features.index.map(
        lambda img_id: os.path.join(train_images_path, f"{img_id}.jpg"))
    test_features['filepath'] = test_features.index.map(
        lambda img_id: os.path.join(test_images_path, f"{img_id}.jpg"))

    # Extract sorted species labels
    species_labels = sorted(train_labels.columns.unique())

    return train_features, test_features, train_labels, species_labels


def split_data(train_features, train_labels, type=None, random_state=42):
    """
    Split data by sites or otherwise.
    Args:
        train_features (pd.DataFrame): DataFrame containing file paths to images.
        train_labels (pd.DataFrame): DataFrame containing one-hot encoded labels.
        type (str): Strategy to split the data. Options are:
            - 'default': Random split with stratification by labels.
            - 'sites': Split based on unique sites.

    Returns:
        X_train, X_val, y_train, y_val (tuple): Training and validation 
            splits for features and labels.
    """
    if type == 'sites': 
        # Get unique sites
        unique_sites = train_features['site'].unique()

        # Split sites into two sets
        train_sites, val_sites = train_test_split(unique_sites, test_size=0.2, 
                                                random_state=random_state)

        # Create boolean masks for training and validation splits
        train_mask = train_features['site'].isin(train_sites)
        val_mask = train_features['site'].isin(val_sites)

        # Apply masks to split the features
        X_train = train_features[train_mask]
        X_val = train_features[val_mask]
        y_train = train_labels[train_mask]
        y_val = train_labels[val_mask]
    
    else: # Split the data (80% training, 20% validation)
        X_train, X_val, y_train, y_val = train_test_split(
            train_features, 
            train_labels, 
            test_size=0.2, 
            stratify=train_labels, 
            random_state=random_state
    )
        
    return X_train, X_val, y_train, y_val


class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, features, 
                 labels=None, 
                 transform=None, device='cpu'):
        self.data = features
        self.label = labels
        self.device = device
        self.transform = transform

    def __getitem__(self, index):
    
        image_id = self.data.index[index]
        image = Image.open(self.data.iloc[index]["filepath"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample = {"image_id": image_id, "image": image}
     
        if self.label is not None:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            if self.device:
                label = label.to(self.device)
            sample["label"] = label

        return sample

    def __len__(self):
        return len(self.data)



def block_timestamp(image):
    """
    Arg: PIL
    """
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    _,height, width = image_tensor.shape
    
    crop_height = int(0.07 * height)  # 5% of the image height
    v = torch.tensor(0)  # Erase value as a tensor
    tensor_output= TF.erase(image_tensor, height - crop_height, 0, crop_height, width, v)
    to_pil = transforms.ToPILImage()
    output = to_pil(tensor_output)
    return output


def get_transforms(config, seed=42):
    """
    Builds torchvision transforms.Compose object based on conf.
    Args: config (dict): Dictionary with transform parameters.
    Returns: transforms.Compose: train_transforms, val_transforms
    """
    src.set_seeds(seed)
    train_transform_list = [
        transforms.Resize(config["transforms"]["resize"]),
    ]

    val_transform_list = [
        transforms.Resize(config["transforms"]["resize"]),
    ]

    # Add custom Lambda transform if specified
    if config["transforms"].get("custom", {}).get("block_timestamp", False):
        train_transform_list.append(transforms.Lambda(block_timestamp))
        val_transform_list.append(transforms.Lambda(block_timestamp))
        

    train_transform_list.extend([
        transforms.RandomHorizontalFlip(p=config["transforms"]["horizontal_flip"]),
        transforms.RandomRotation(config["transforms"]["rotate"]),
        transforms.ColorJitter(
            brightness=config["transforms"]["jitter"]["brightness"],
            contrast=config["transforms"]["jitter"]["contrast"],
            saturation=config["transforms"]["jitter"]["saturation"],
            hue=config["transforms"]["jitter"]["hue"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Convert to tensor and normalize
    val_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_transforms =  transforms.Compose(train_transform_list)
    val_transforms = transforms.Compose(val_transform_list)

    return train_transforms, val_transforms


########## currently not in use #########

def block_timestamp_and_resize(image, target_size=(224, 224)):
    """
    Block timestamp and resize image with matching padding color.
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Desired output size (height, width)
    Returns:
        PIL.Image: Processed image with blocked timestamp and padding
    """
    # Get edge colors first (before any modifications)
    edge_pixels = np.array(image)
    edge_pixels = np.concatenate([
        edge_pixels[0],  # top edge
        edge_pixels[-1],  # bottom edge
        edge_pixels[:, 0],  # left edge
        edge_pixels[:, -1]  # right edge
    ])
    mean_color = edge_pixels.mean(axis=0)
    
    # Add random variation for training
    if random.random() < 0.5:  # 50% chance of variation
        random_variation = np.random.randint(-20, 20, size=3)
        padding_color = tuple(np.clip(mean_color + random_variation, 0, 255).astype(int))
    else:
        padding_color = tuple(map(int, mean_color))
    
    # Convert to tensor to block timestamp
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    _, height, width = image_tensor.shape
    
    # Convert padding color to tensor format (0-1 range)
    erase_value = torch.tensor([c/255.0 for c in padding_color]).view(3, 1, 1)
    
    # Block timestamp
    crop_height = int(0.07 * height)
    tensor_output = TF.erase(
        image_tensor, 
        height - crop_height, 0, 
        crop_height, width, 
        erase_value
    )
    
    # Convert back to PIL
    to_pil = transforms.ToPILImage()
    blocked_image = to_pil(tensor_output)
    
    # Calculate new dimensions for resizing
    current_width, current_height = blocked_image.size
    aspect_ratio = current_width / current_height
    target_aspect_ratio = target_size[0] / target_size[1]
    
    if aspect_ratio > target_aspect_ratio:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    
    # Resize image
    resized_image = blocked_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    # Create padded image with same color as timestamp block
    padded_image = Image.new('RGB', target_size, padding_color)
    
    # Calculate offsets for random positioning during training
    max_offset_x = max(0, target_size[0] - new_width)
    max_offset_y = max(0, target_size[1] - new_height)
    
    # Random position for training, center for validation
    if max_offset_x > 0 or max_offset_y > 0:
        offset_x = random.randint(0, max_offset_x)
        offset_y = random.randint(0, max_offset_y)
    else:
        offset_x = (target_size[0] - new_width) // 2
        offset_y = (target_size[1] - new_height) // 2
    
    # Paste resized image onto padded background
    padded_image.paste(resized_image, (offset_x, offset_y))
    
    return padded_image

# Validation transforms
def validation_block_and_resize(image, target_size=(224, 224)):
    """
    Consistent version for validation - no random variations
    """
    # Get edge colors
    edge_pixels = np.array(image)
    edge_pixels = np.concatenate([
        edge_pixels[0], edge_pixels[-1], 
        edge_pixels[:, 0], edge_pixels[:, -1]
    ])
    padding_color = tuple(map(int, edge_pixels.mean(axis=0)))
    
    # Block timestamp
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    _, height, width = image_tensor.shape
    
    erase_value = torch.tensor([c/255.0 for c in padding_color]).view(3, 1, 1)
    crop_height = int(0.07 * height)
    tensor_output = TF.erase(
        image_tensor, 
        height - crop_height, 0, 
        crop_height, width, 
        erase_value
    )
    
    to_pil = transforms.ToPILImage()
    blocked_image = to_pil(tensor_output)
    
    # Resize with aspect ratio preservation
    current_width, current_height = blocked_image.size
    aspect_ratio = current_width / current_height
    target_aspect_ratio = target_size[0] / target_size[1]
    
    if aspect_ratio > target_aspect_ratio:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    
    resized_image = blocked_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    # Create padded image
    padded_image = Image.new('RGB', target_size, padding_color)
    
    # Center the image
    offset_x = (target_size[0] - new_width) // 2
    offset_y = (target_size[1] - new_height) // 2
    
    padded_image.paste(resized_image, (offset_x, offset_y))
    
    return padded_image


def preserve_aspect_ratio_resize(image, target_size):
    """
    Resize image to target size while maintaining aspect ratio and padding if necessary.
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Desired output size (height, width)
        
    Returns:
        PIL.Image: Resized and padded image
    """
    target_height, target_width = target_size
    
    # Get current image size
    width, height = image.size
    
    # Calculate aspect ratios
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    
    # Calculate aspect ratios
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    
    # Calculate new dimensions
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target - scale based on width
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Image is taller than target - scale based on height
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    
    # Resize image while maintaining aspect ratio
    resized_image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    # Create new image with padding
    padded_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    
    # Calculate padding
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    # Paste resized image onto padded background
    padded_image.paste(resized_image, (pad_left, pad_top))
    
    return padded_image




