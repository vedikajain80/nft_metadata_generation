import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_images_and_labels(metadata_file, img_dir, img_size=(224, 224)):
    metadata = pd.read_csv(metadata_file)
    num_samples = len(metadata)
    images = np.zeros((num_samples, img_size[0], img_size[1], 3))
    labels = []

    for i, row in metadata.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array) 
        images[i] = img_array
        labels.append(row['background'])

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(integer_labels)

    return images, one_hot_labels


def create_model(num_classes):
    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom classification layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Add dropout layer
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def train_model():
    train_dir = 'train'
    val_dir = 'val'
    num_classes = 8
    epochs = 5
    learning_rate = 0.0001
    model_save_path = 'best_model.h5'

    # Load preprocessed data
    train_metadata_file = os.path.join(train_dir, 'metadata.csv')
    train_img_dir = os.path.join(train_dir, 'images')
    val_metadata_file = os.path.join(val_dir, 'metadata.csv')
    val_img_dir = os.path.join(val_dir, 'images')

    X_train, y_train = preprocess_images_and_labels(train_metadata_file, train_img_dir)
    X_val, y_val = preprocess_images_and_labels(val_metadata_file, val_img_dir)

    # Create and compile the model
    model = create_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Set up the checkpoint callback to save the best model
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    print("begin training")
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint])

    return history


def plot_history(history):
    print("plotting")
    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    # Train the model and get the history object
    history = train_model()

    # Plot the training and validation losses and accuracies
    plot_history(history)
    print("finished")
