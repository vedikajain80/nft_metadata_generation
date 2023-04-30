# NFT Metadata Generator

This project is a proof-of-concept AI-driven service that can identify and classify unique attributes in visual images as part of an existing NFT collection.

## Setup

1. Install Python 3.x from https://www.python.org/downloads/
2. Clone this repository: `git clone https://github.com/yourusername/nft_metadata_generator.git`
3. Change directory: `cd nft_metadata_generator`
4. Create a virtual environment: `python3 -m venv venv`
5. Activate the virtual environment:
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `.\venv\Scripts\activate`
6. Install dependencies: `pip install -r requirements.txt`

-- This setup process for virtual env and dependencies may look different if using Mac M1 chip.

## Usage

1. Fetch the metadata and images: `python3 fetch_metadata.py`
2. Prepare the dataset: `python3 prepare_dataset.py`
3. Train the model: `python3 train_model.py`
4. Test the mode: `python3 test_model.py`
4. Predict attributes and generate JSON (3 as user input when asked):`python3 predict.py`

## Predict attributes of an image using saved model - FVCKCRYSTAL/BAYC

1. generate image file for specific contract address: `python3 generate_test_set.py` 
2. Predict attributes and generate JSON:`python3 predict.py`

## Results 

The accuracy and loss graphs for training and validation sets can be found in the results folder

## Places for improvement 

downloading all the images 