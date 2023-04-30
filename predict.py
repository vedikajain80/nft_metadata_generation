import json
import os
import numpy as np
import random
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_random_test_image_path(test_dir):
    test_metadata = pd.read_csv(os.path.join(test_dir, 'metadata.csv'))
    random_image_filename = random.choice(test_metadata['filename'].tolist())
    return os.path.join(test_dir, 'images', random_image_filename)

def predict_attributes(model, image_path, label_encoders):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_attributes = {}
    for i, attr_name in enumerate(label_encoders.keys()):
        predicted_index = np.argmax(predictions[i])
        try:
            predicted_value = label_encoders[attr_name].inverse_transform([predicted_index])[0]
            predicted_attributes[attr_name] = predicted_value
        except ValueError:
            print(f"Warning: Model predicted an unseen label for attribute '{attr_name}'. Setting the value to 'Unknown'.")
            predicted_attributes[attr_name] = "Unknown"

    return predicted_attributes

def generate_erc721_metadata(predicted_attributes, image_url):
    metadata = {
        "name": "Your Token Name",
        "description": "Your Token Description",
        "image": image_url,
        "attributes": []
    }

    for attr_name, attr_value in predicted_attributes.items():
        if(attr_value != "absent"):
            metadata["attributes"].append({
                "trait_type": attr_name,
                "value": attr_value
            })

    return metadata

def main():
    test_dir = 'test'

    collection = input("1 for FVCKCRYSTALS, 2 for BAYC, 3 for collection trained from scratch: ").strip()

    # default is BAYC
    model_path = 'saved_models/best_model_BAYC.h5'
    saved_label_encoders = 'saved_models/label_encoders_BAYC.pkl'

    # FVCKCRYSTALS
    if(collection == "1"):
        model_path = 'saved_models/best_model_CRYSTALS.h5'
        saved_label_encoders = 'saved_models/label_encoders_CRYSTALS.pkl'

    # collection trained from scratch
    if(collection == "3"):
        model_path = 'best_model.h5'
        saved_label_encoders = 'label_encoders.pkl'

    # Get a random image path from the test set
    img_path = get_random_test_image_path(test_dir)

    # Extract image filename without extension
    image_filename = os.path.splitext(os.path.basename(img_path))[0]

    print("image_filename:", image_filename)

    # Load the model
    model = load_model(model_path)
    
    with open(saved_label_encoders, 'rb') as f:
        label_encoders = pickle.load(f)

    # Predict attributes
    predicted_attributes = predict_attributes(model, img_path, label_encoders)

    # Print predicted_attributes to check for issues
    print("Predicted attributes:", predicted_attributes)

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
