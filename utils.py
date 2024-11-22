"""
Helper Functions
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split

def verify_data(train_features, test_features, train_labels):
    print("train_features")
    print(train_features.head())

    print("\n test_features")
    print(test_features.head())

    print("\n train_labels")
    print(train_labels.head())

    print("\n Species Distribution by Count and Percentage")
    print(train_labels.sum().sort_values(ascending=False))
    print(train_labels.sum().divide(train_labels.shape[0]).sort_values(ascending=False))

def plot_species_grid(train_features, 
                      train_labels, 
                      species_labels, 
                      random_state=42):
    """
    Plots a grid of images, one for each species label.
    Args:
        train_features (pd.DataFrame): DataFrame containing file paths to images.
        train_labels (pd.DataFrame): DataFrame containing one-hot encoded labels.
        species_labels (list): List of species names corresponding to the columns in train_labels.
        random_state (int): Seed for reproducibility when sampling images.
    """

    # Create a grid with 8 positions (4 rows x 2 columns)
    # one for each label (7 species, plus blanks)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))

    # Iterate through each species and plot one image per species
    for species, ax in zip(species_labels, axes.flat):
        # Sample an image ID for the given species
        img_id = (
            train_labels[train_labels.loc[:, species] == 1]
            .sample(1, random_state=random_state)
            .index[0]
        )

        # Read the image file
        img = mpimg.imread(train_features.loc[img_id].filepath)

        # Display the image in the subplot
        ax.imshow(img)
        ax.set_title(f"{img_id} | {species}")
        ax.axis("off")  # Optional: Hide axes for cleaner visuals

    # Show the plot
    plt.tight_layout()
    plt.show()

def verify_splits(X_train, y_train, X_val,  y_val):
    print("X_train")
    print(X_train.head())

    print("\n y_train")
    print(y_train.head())

    print("\n X_val")
    print(X_val.head())

    print("\n y_val")
    print(y_val.head())

    print("\n Data Shape")
    print("Train: ", X_train.shape, y_train.shape, "Validate: ", X_val.shape, y_val.shape)

    split_pcts = pd.DataFrame(
    {
        "train": y_train.idxmax(axis=1).value_counts(normalize=True),
        "eval": y_val.idxmax(axis=1).value_counts(normalize=True),
    }
    )
    print("\n Species percentages by split")
    print(split_pcts.fillna(0))

def split_data(train_features, train_labels, type=None):
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
                                                random_state=42)

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
            random_state=42
    )
        
    return X_train, X_val, y_train, y_val

# def split_data(train_features, train_labels, split_strategy="default"):
#     """
#     Split data based on the specified strategy.
    
#     Args:
#         train_features (pd.DataFrame): DataFrame containing file paths to images.
#         train_labels (pd.DataFrame): DataFrame containing one-hot encoded labels.
#         split_strategy (str): Strategy to split the data. Options are:
#             - 'default': Random split with stratification by labels.
#             - 'sites': Split based on unique sites.
#             - 'balanced_sites': Balanced split by site distribution.
    
#     Returns:
#         X_train, X_val, y_train, y_val (tuple): Training and validation splits for features and labels.
#     """
#     print("test")
#     if split_strategy == "sites":
#         # Split by unique sites
#         unique_sites = train_features['site'].unique()
#         train_sites, val_sites = train_test_split(unique_sites, test_size=0.2, random_state=42)
        
#     elif split_strategy == "balanced_sites":
#         # Merge features and labels to associate sites with animal counts
#         site_animal_distribution = train_features.merge(
#             train_labels, left_index=True, right_index=True
#         ).drop(columns=["filepath"]).groupby('site').sum()

#         # Add a total column for sorting
#         site_animal_distribution['total'] = site_animal_distribution.sum(axis=1)
#         sorted_sites = site_animal_distribution.sort_values(by='total', ascending=False)

#         # Initialize splits
#         train_sites, val_sites = [], []
#         train_distribution = pd.Series(0, index=train_labels.columns)
#         val_distribution = pd.Series(0, index=train_labels.columns)
#         train_count, val_count = 0, 0

#         # Target counts for samples
#         total_samples = len(train_features)
#         train_target = int(0.8 * total_samples)
#         val_target = total_samples - train_target

#         # Iteratively split sites
#         for site, row in sorted_sites.iterrows():
#             # Calculate the number of samples for this site
#             site_sample_count = site_animal_distribution.loc[site, 'total']

#             # Check which split to assign based on current distributions and target counts
#             if (
#                 train_count + site_sample_count <= train_target
#                 and (train_distribution + row).std() <= (val_distribution + row).std()
#             ):
#                 train_sites.append(site)
#                 train_distribution += row
#                 train_count += site_sample_count
#             else:
#                 val_sites.append(site)
#                 val_distribution += row
#                 val_count += site_sample_count

#     else:  # Default random split
#         return train_test_split(
#             train_features, train_labels, test_size=0.2, stratify=train_labels, random_state=42
#         )
    
#     # Create masks and apply splits
#     train_mask = train_features['site'].isin(train_sites)
#     val_mask = train_features['site'].isin(val_sites)

#     X_train = train_features[train_mask]
#     X_val = train_features[val_mask]
#     y_train = train_labels.loc[X_train.index]
#     y_val = train_labels.loc[X_val.index]

#     return X_train, X_val, y_train, y_val

