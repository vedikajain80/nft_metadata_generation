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

1. Fetch the metadata and images: `python fetch_metadata.py`
2. Prepare the dataset: `python prepare_dataset.py`
3. Train the model: `python train_model.py`