import numpy as np
import pandas as pd
import json
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from train_model import preprocess_images_and_labels

# Load the label encoders
saved_label_encoders = 'label_encoders.pkl'  # Update the path based on the desired collection
with open(saved_label_encoders, 'rb') as f:
    label_encoders = pickle.load(f)

saved_classes = 'num_classes_train.pkl'  # Update the path based on the desired collection
with open(saved_classes, 'rb') as f:
    num_classes_train = pickle.load(f)

# Load the test metadata
test_dir = 'test'
test_metadata_file = os.path.join(test_dir, 'metadata.csv')
test_img_dir = os.path.join(test_dir, 'images')

# Preprocess the test images and labels
test_images, test_one_hot_labels_dict = preprocess_images_and_labels(
    test_metadata_file, test_img_dir, label_encoders, num_classes_train
)

# Load the trained model
model_path = 'best_model.h5'  # Update the model path based on the desired collection
model = load_model(model_path)

# Evaluate the model on the test set
test_pred = model.predict(test_images)

# Print evaluation metrics for each attribute
for i, attr_name in enumerate(label_encoders.keys()):
    y_true = label_encoders[attr_name].inverse_transform(np.argmax(test_one_hot_labels_dict[attr_name], axis=1))
    y_pred = label_encoders[attr_name].inverse_transform(np.argmax(test_pred[i], axis=1))
    with open(f"{os.path.splitext(os.path.basename(model_path))[0]}_evaluation.txt", "a") as eval_file:
        eval_file.write(f"Evaluation for attribute: {attr_name}\n")
        eval_file.write(classification_report(y_true, y_pred, zero_division=0))
        eval_file.write("\n\n")


