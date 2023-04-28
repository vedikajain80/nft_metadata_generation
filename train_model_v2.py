import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_images_and_labels(metadata_file, img_dir, label_encoders=None, img_size=(224, 224), num_classes_train=None):
    metadata = pd.read_csv(metadata_file)
    num_samples = len(metadata)
    images = np.zeros((num_samples, img_size[0], img_size[1], 3))
    all_attribute_names = set()
    attributes_dict = {}

    if num_classes_train is None:
        for i, row in metadata.iterrows():
            attributes = json.loads(row['attributes'])
            all_attribute_names.update(attributes.keys())
    else:
        all_attribute_names = num_classes_train.keys()

    
    for attr_name in all_attribute_names:
        attributes_dict[attr_name] = []


    for i, row in metadata.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        images[i] = img_array
        
        attributes = json.loads(row['attributes'])
        for attr_name in all_attribute_names:
            if attr_name in attributes:
                attributes_dict[attr_name].append(attributes[attr_name])
            else:
                attributes_dict[attr_name].append("absent")

    if label_encoders is None:
        label_encoders = {attr_name: LabelEncoder() for attr_name in all_attribute_names}
        for attr_name, attr_values in attributes_dict.items():
            label_encoders[attr_name].fit(attr_values)
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)

    one_hot_labels_dict = {}
    num_classes = {}

    for attr_name, attr_values in attributes_dict.items():
        integer_labels = label_encoders[attr_name].transform(attr_values)
        if num_classes_train is not None:
            num_classes[attr_name] = num_classes_train[attr_name]
            one_hot_labels = to_categorical(integer_labels, num_classes[attr_name])
        else:
            one_hot_labels = to_categorical(integer_labels)
            num_classes[attr_name] = len(np.unique(integer_labels))
        one_hot_labels_dict[attr_name] = one_hot_labels

    return images, one_hot_labels_dict, num_classes, label_encoders


def create_model(num_classes_dict):
    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom classification layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Add dropout layer

    outputs = []
    for attr_name, num_classes in num_classes_dict.items():
        output = Dense(num_classes, activation='softmax', name=f'{attr_name}_output')(x)
        outputs.append(output)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def train_model():
    train_dir = 'train'
    val_dir = 'val'
    epochs = 25
    learning_rate = 0.0001
    model_save_path = 'best_model.h5'

    # Load preprocessed data
    train_metadata_file = os.path.join(train_dir, 'metadata.csv')
    train_img_dir = os.path.join(train_dir, 'images')
    val_metadata_file = os.path.join(val_dir, 'metadata.csv')
    val_img_dir = os.path.join(val_dir, 'images')

    X_train, y_train_dict, num_classes_train, label_encoders = preprocess_images_and_labels(train_metadata_file, train_img_dir)
    X_val, y_val_dict, _, _ = preprocess_images_and_labels(val_metadata_file, val_img_dir, label_encoders, num_classes_train=num_classes_train)

    # Create and compile the model
    model = create_model(num_classes_train)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss={f'{attr_name}_output': 'categorical_crossentropy' for attr_name in num_classes_train},
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True)

    # Train the model
    print("begin training")
    history = model.fit(X_train, {f'{attr_name}_output': y_train_dict[attr_name] for attr_name in y_train_dict},
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_val, {f'{attr_name}_output': y_val_dict[attr_name] for attr_name in y_val_dict}),
                        callbacks=[checkpoint])

    return history, num_classes_train


def plot_history(history, num_classes):
    print("plotting")
    # Plot accuracy for background_output
    for attr_name in num_classes:
        # Plot accuracy
        plt.figure()
        plt.plot(history.history[f'{attr_name}_output_accuracy'], label=f'Training {attr_name} Accuracy')
        plt.plot(history.history[f'val_{attr_name}_output_accuracy'], label=f'Validation {attr_name} Accuracy')
        plt.title(f'{attr_name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.figure()
        plt.plot(history.history[f'{attr_name}_output_loss'], label=f'Training {attr_name} Loss')
        plt.plot(history.history[f'val_{attr_name}_output_loss'], label=f'Validation {attr_name} Loss')
        plt.title(f'{attr_name} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    plt.show()

if __name__ == '__main__':
    # Train the model and get the history object
    history, num_classes = train_model()

    # Plot the training and validation losses and accuracies
    plot_history(history, num_classes)
