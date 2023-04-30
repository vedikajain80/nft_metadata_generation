import shutil
import requests
import json
import pandas as pd
import os
import time
import math

# Create dataset and images directories if they don't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

if not os.path.exists(os.path.join('dataset', 'images')):
    os.makedirs(os.path.join('dataset', 'images'))

if not os.path.exists(os.path.join('dataset', 'jsons')):
    os.makedirs(os.path.join('dataset', 'jsons'))

# Fetch assets from NFTPort API
def fetch_assets(api_key, contract_address, chain="ethereum", page_size=50, count=10000):
    all_assets = []

    # Fetch the first page to get the total count of assets
    url = f"https://api.nftport.xyz/v0/nfts/{contract_address}?chain={chain}&page_number=1&page_size={page_size}&include=metadata&refresh_metadata=false"

    headers = {
        "accept": "application/json",
        "Authorization": api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        total_count = min(count, data['total'])
        all_assets.extend(data['nfts'])

        #Calculate the total number of pages
        total_pages = math.ceil(total_count / page_size)

        with open(os.path.join('dataset', 'jsons', f'page_1.json'), 'w') as outfile:
            json.dump(data, outfile)

        print(f"fetching {total_pages} pages")

        # Fetch the remaining pages
        for page_number in range(2, total_pages + 1):
            url = f"https://api.nftport.xyz/v0/nfts/{contract_address}?chain=ethereum&page_number={page_number}&page_size={page_size}&include=metadata&refresh_metadata=false"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                all_assets.extend(data['nfts'])
                with open(os.path.join('dataset', 'jsons', f'page_{page_number}.json'), 'w') as outfile:
                    json.dump(data, outfile)
                print(f"page number: {page_number}")

            else:
                print(f"Error fetching assets (page {page_number}): {response.status_code}")

    else:
        print(f"Error fetching assets: {response.status_code}")

    return all_assets

# Extract relevant information
def extract_metadata(nfts):
    print("extracting metadata")
    metadata_list = []
    for nft in nfts:
        token_id = nft['token_id']
        filename = f"{token_id}.png"
        traits = {trait['trait_type']: trait['value'] for trait in nft['metadata']['attributes']}
        file_path = os.path.join('dataset', 'images', filename)

        # Download and save image
        image_url = nft['cached_file_url']
        if image_url is not None:
            response = requests.get(image_url, stream=True)
            with open(os.path.join('dataset', 'images', filename), 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            metadata_list.append({'token_id': token_id, 'filename': filename, 'attributes': json.dumps(traits)})
        else:
            print(f"Error: Image URL is missing for token_id {token_id}")
            
    return metadata_list


if __name__ == '__main__':
    contract_address = "0x7AfEdA4c714e1C0A2a1248332c100924506aC8e6"  
    api_key = "ac5d347e-c8fb-4ab6-9455-2c6823776e3a"
    
    assets = fetch_assets(api_key, contract_address)
    metadata_list = extract_metadata(assets)
    
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(os.path.join('dataset', 'metadata.csv'), index=False)

