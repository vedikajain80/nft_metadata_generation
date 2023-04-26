import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def preprocess_images_and_labels(metadata_file, img_dir, img_size=(224, 224)):
    metadata = pd.read_csv(metadata_file)
    num_samples = len(metadata)
    images = np.zeros((num_samples, img_size[0], img_size[1], 3))
    labels = []

    for i, row in metadata.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        images[i] = img_array
        labels.append(row['background'])  # Changed this line to match the new column name

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(integer_labels)

    return images, one_hot_labels