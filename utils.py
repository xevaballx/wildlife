"""
Helper Functions
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

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