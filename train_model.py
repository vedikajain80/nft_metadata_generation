import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import json
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
    backgrounds = []
    eyes = []

    for i, row in metadata.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array) 
        images[i] = img_array

        attributes = json.loads(row['attributes'])
        backgrounds.append(attributes['Background'])
        eyes.append(attributes['Eyes'])

    label_encoder = LabelEncoder()

    # Encode backgrounds
    integer_labels_bg = label_encoder.fit_transform(backgrounds)
    one_hot_labels_bg = to_categorical(integer_labels_bg)

    # Encode eyes
    integer_labels_eyes = label_encoder.fit_transform(eyes)
    one_hot_labels_eyes = to_categorical(integer_labels_eyes)

    return images, one_hot_labels_bg, one_hot_labels_eyes


def create_model(num_classes_bg, num_classes_eyes):
    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom classification layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Add dropout layer
    background_output = Dense(num_classes_bg, activation='softmax', name='background_output')(x)
    eyes_output = Dense(num_classes_eyes, activation='softmax', name='eyes_output')(x)

    model = Model(inputs=base_model.input, outputs=[background_output, eyes_output])
    return model


def train_model():
    train_dir = 'train'
    val_dir = 'val'
    epochs = 10
    learning_rate = 0.0001
    model_save_path = 'best_model.h5'
    num_classes_bg = 8
    num_classes_eyes = 23

    # Load preprocessed data
    train_metadata_file = os.path.join(train_dir, 'metadata.csv')
    train_img_dir = os.path.join(train_dir, 'images')
    val_metadata_file = os.path.join(val_dir, 'metadata.csv')
    val_img_dir = os.path.join(val_dir, 'images')

    X_train, y_train_bg, y_train_eyes = preprocess_images_and_labels(train_metadata_file, train_img_dir)
    X_val, y_val_bg, y_val_eyes = preprocess_images_and_labels(val_metadata_file, val_img_dir)

    # Create and compile the model
    model = create_model(num_classes_bg, num_classes_eyes)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss={'background_output': 'categorical_crossentropy', 'eyes_output': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True)

    # Train the model
    print("begin training")
    history = model.fit(X_train, {'background_output': y_train_bg, 'eyes_output': y_train_eyes},
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_val, {'background_output': y_val_bg, 'eyes_output': y_val_eyes}),
                        callbacks=[checkpoint])

    return history


def plot_history(history):
    print("plotting")
    # Plot accuracy for background_output
    plt.figure()
    plt.plot(history.history['background_output_accuracy'], label='Training Background Accuracy')
    plt.plot(history.history['val_background_output_accuracy'], label='Validation Background Accuracy')
    plt.title('Background Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot accuracy for eyes_output
    plt.figure()
    plt.plot(history.history['eyes_output_accuracy'], label='Training Eyes Accuracy')
    plt.plot(history.history['val_eyes_output_accuracy'], label='Validation Eyes Accuracy')
    plt.title('Eyes Accuracy')
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
