import os
import shutil
import random
import pandas as pd
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def make_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_label_encoders_and_num_classes(dataset_dir):
    metadata = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'))
    all_attribute_names = set()

    for i, row in metadata.iterrows():
        attributes = json.loads(row['attributes'])
        all_attribute_names.update(attributes.keys())

    label_encoders = {attr_name: LabelEncoder() for attr_name in all_attribute_names}

    num_classes_train = {}
    for attr_name in all_attribute_names:
        attr_values = [json.loads(row['attributes']).get(attr_name, "absent") for _, row in metadata.iterrows()]
        label_encoders[attr_name].fit(attr_values)
        num_classes_train[attr_name] = len(label_encoders[attr_name].classes_)

    return label_encoders, num_classes_train

def split_dataset(dataset_dir, train_dir, val_dir, test_dir, label_encoders, num_classes_train, split_ratio=(0.7, 0.2, 0.1)):
    metadata = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'))

    # Shuffle the dataset
    metadata = metadata.sample(frac=1).reset_index(drop=True)

    # Find the indices of rows that have at least one instance of every attribute class
    train_indices = []
    for attr_name, num_classes in num_classes_train.items():
        for class_index in range(num_classes):
            attr_class_rows = metadata[metadata['attributes'].apply(lambda x: json.loads(x).get(attr_name)) == label_encoders[attr_name].inverse_transform([class_index])[0]]
            if not attr_class_rows.empty:
                train_indices.append(random.choice(attr_class_rows.index))

    # Add the rows with at least one instance of every attribute class to the training set
    train_metadata = metadata.loc[train_indices]
    train_metadata.to_csv(os.path.join(train_dir, 'metadata.csv'), index=False)

    # Split the metadata and save to respective directories
    total_samples = len(metadata)
    val_samples = int(total_samples * split_ratio[1])
    test_samples = int(total_samples * split_ratio[2])

    metadata[:val_samples].to_csv(os.path.join(val_dir, 'metadata.csv'), index=False)
    metadata[val_samples:val_samples + test_samples].to_csv(os.path.join(test_dir, 'metadata.csv'), index=False)
    concatenated_metadata = pd.concat([train_metadata, metadata[val_samples + test_samples:]], axis=0)

    # Save the concatenated dataframe to a CSV file
    concatenated_metadata.to_csv(os.path.join(train_dir, 'metadata.csv'), index=False)

    # Copy images to respective directories
    for idx, row in metadata.iterrows():
        src = os.path.join(dataset_dir, 'images', row['filename'])
        if idx < val_samples:
            dest = os.path.join(val_dir, 'images', row['filename'])
        elif idx < val_samples + test_samples:
            dest = os.path.join(test_dir, 'images', row['filename'])
        else:
            dest = os.path.join(train_dir, 'images', row['filename'])
        shutil.copy(src, dest)

    # Copy images with at least one instance of every attribute class to train directory
    for idx, row in train_metadata.iterrows():
        src = os.path.join(dataset_dir, 'images', row['filename'])
        dest = os.path.join(train_dir, 'images', row['filename'])
        shutil.copy(src, dest)

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    dataset_dir = 'dataset'
    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'

    make_directory_if_not_exists(train_dir)
    make_directory_if_not_exists(os.path.join(train_dir, 'images'))
    make_directory_if_not_exists(val_dir)
    make_directory_if_not_exists(os.path.join(val_dir, 'images'))
    make_directory_if_not_exists(test_dir)
    make_directory_if_not_exists(os.path.join(test_dir, 'images'))

    label_encoders, num_classes_train = create_label_encoders_and_num_classes(dataset_dir)
    split_dataset(dataset_dir, train_dir, val_dir, test_dir, label_encoders, num_classes_train)

    save_pickle(label_encoders, 'label_encoders.pkl')
    save_pickle(num_classes_train, 'num_classes_train.pkl')