import json
import os
import numpy as np
import random
import pandas as pd
from train_model import preprocess_images_and_labels
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder

def get_random_test_image_path(test_dir):
    test_metadata = pd.read_csv(os.path.join(test_dir, 'metadata.csv'))
    random_image_filename = random.choice(test_metadata['filename'].tolist())
    return os.path.join(test_dir, 'images', random_image_filename)

def predict_attributes(model, image_path, num_classes, label_encoders):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    print("Predictions:", predictions)

    predicted_attributes = {}
    for i, attr_name in enumerate(label_encoders.keys()):
        predicted_index = np.argmax(predictions[i])
        predicted_attributes[attr_name] = label_encoders[attr_name].inverse_transform([predicted_index])

    return predicted_attributes

def generate_erc721_metadata(predicted_attributes, image_url):
    metadata = {
        "name": "Your Token Name",
        "description": "Your Token Description",
        "image": image_url,
        "attributes": []
    }

    for attr_name, attr_value in predicted_attributes.items():
        metadata["attributes"].append({
            "trait_type": attr_name,
            "value": attr_value.tolist()[0]
        })

    return metadata

def main():
    model_path = 'best_without_absent_model.h5'
    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'
    
    # Get a random image path from the test set
    img_path = get_random_test_image_path(test_dir)

    train_metadata_file = os.path.join(train_dir, 'metadata.csv')
    train_img_dir = os.path.join(train_dir, 'images')
    train_metadata = pd.read_csv(train_metadata_file)
    attributes_dict = {}

    for _, row in train_metadata.iterrows():
        attributes = json.loads(row['attributes'])
        for attr_name, attr_value in attributes.items():
            if attr_name not in attributes_dict:
                attributes_dict[attr_name] = []
            attributes_dict[attr_name].append(attr_value)

    _, one_hot_labels_dict, num_classes = preprocess_images_and_labels(train_metadata_file, train_img_dir)

    label_encoders = {}
    for attr_name in num_classes.keys():
        label_encoder = LabelEncoder()
        label_encoder.fit(attributes_dict[attr_name])
        label_encoders[attr_name] = label_encoder

    # Load the model
    model = load_model(model_path)

    # Predict attributes
    predicted_attributes = predict_attributes(model, img_path, num_classes, label_encoders)

    # Print predicted_attributes to check for issues
    print("Predicted attributes:", predicted_attributes)

    # Extract image filename without extension
    image_filename = os.path.splitext(os.path.basename(img_path))[0]

    print("image_filename:", image_filename)

    # Generate ERC721 metadata
    image_url = "https://yourdomain.com/path/to/image.jpg"
    metadata = generate_erc721_metadata(predicted_attributes, image_url)
    print(json.dumps(metadata, indent=2))

    # Save metadata to a file with the image name included
    metadata_filename = f"{image_filename}_metadata.json"
    with open(metadata_filename, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

if __name__ == "__main__":
    main()
