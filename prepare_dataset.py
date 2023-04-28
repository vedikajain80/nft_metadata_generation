import os
import shutil
import random
import pandas as pd
import json

def make_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset(dataset_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.2, 0.1)):
    # Read metadata.csv
    metadata = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'))

    # Shuffle the dataset
    metadata = metadata.sample(frac=1).reset_index(drop=True)

    # Calculate the number of samples for each split
    total_samples = len(metadata)
    train_samples = int(total_samples * split_ratio[0])
    val_samples = int(total_samples * split_ratio[1])

    # Create directories if they do not exist
    make_directory_if_not_exists(train_dir)
    make_directory_if_not_exists(os.path.join(train_dir, 'images'))
    make_directory_if_not_exists(val_dir)
    make_directory_if_not_exists(os.path.join(val_dir, 'images'))
    make_directory_if_not_exists(test_dir)
    make_directory_if_not_exists(os.path.join(test_dir, 'images'))

    # Split metadata and save to respective directories
    metadata[:train_samples].to_csv(os.path.join(train_dir, 'metadata.csv'), index=False)
    metadata[train_samples:train_samples + val_samples].to_csv(os.path.join(val_dir, 'metadata.csv'), index=False)
    metadata[train_samples + val_samples:].to_csv(os.path.join(test_dir, 'metadata.csv'), index=False)

    # Move images to respective directories
    for idx, row in metadata.iterrows():
        src = os.path.join(dataset_dir, 'images', row['filename'])
        if idx < train_samples:
            dest = os.path.join(train_dir, 'images', row['filename'])
        elif idx < train_samples + val_samples:
            dest = os.path.join(val_dir, 'images', row['filename'])
        else:
            dest = os.path.join(test_dir, 'images', row['filename'])
        shutil.copy(src, dest)

if __name__ == '__main__':
    dataset_dir = 'dataset'
    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'

    split_dataset(dataset_dir, train_dir, val_dir, test_dir)
