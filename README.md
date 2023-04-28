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

## Usage

1. Fetch the metadata and images: `python3 fetch_metadata.py`
2. Prepare the dataset: `python3 prepare_dataset.py`
3. Train the model: `python3 train_model.py`
4. Predict attributes and generate JSON:`python3 predict.py`

## Test Model against BAYC collection
1. unzip test_BAYC.zip folder to get images to test: `unzip test_BAYC.zip`
2. Predict attributes and generate JSON:`python3 predict.py`

## Test Model against FVCKCRYSTAL
1. download images to test file: `python3 generate_test_set.py`
2. go to predict.py and change lines 54-56 to 

   ```
   model_path = 'best_model_crystals_v2.h5'
   test_dir = 'test_Crystals'
   saved_label_encoders = 'label_encoders_Crystals.pkl'
   ```

2. Predict attributes and generate JSON:`python3 predict.py`
