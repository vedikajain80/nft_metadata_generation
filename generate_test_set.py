import os
import json
import math
import shutil
import requests
import pandas as pd

def fetch_assets(api_key, contract_address, chain="ethereum", page_size=50, page_number=1):
    all_assets = []

    # Fetch the first page only
    url = f"https://api.nftport.xyz/v0/nfts/{contract_address}?chain={chain}&page_number={page_number}&page_size={page_size}&include=metadata&refresh_metadata=false"

    headers = {
        "accept": "application/json",
        "Authorization": api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        all_assets.extend(data['nfts'])
        print(f"Fetched one page of assets")

    else:
        print(f"Error fetching assets: {response.status_code}")

    return all_assets

def extract_metadata(nfts):
    print("extracting metadata")
    metadata_list = []
    for nft in nfts:
        token_id = nft['token_id']
        filename = f"{token_id}.png"
        traits = {trait['trait_type']: trait['value'] for trait in nft['metadata']['attributes']}
        metadata_list.append({'token_id': token_id, 'filename': filename, 'attributes': json.dumps(traits)})

        # Download and save image
        image_url = nft['cached_file_url']
        response = requests.get(image_url, stream=True)
        with open(os.path.join('test', 'images', filename), 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
            
    return metadata_list

if __name__ == '__main__':
    if not os.path.exists('test'):
        os.makedirs('test')

    if not os.path.exists(os.path.join('test', 'images')):
        os.makedirs(os.path.join('test', 'images'))

    # default contract address is BAYC
    contract_address = "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d"

    collection = input("1 for FVCKCRYSTALS, 2 for BAYC: ").strip()

    if(collection == "1"):
        contract_address = "0x7AfEdA4c714e1C0A2a1248332c100924506aC8e6"

    api_key = "ac5d347e-c8fb-4ab6-9455-2c6823776e3a"
    
    assets = fetch_assets(api_key, contract_address)

    # metadate saved for user to easily look up token id attributes and for randomizing image selection
    metadata_list = extract_metadata(assets)
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(os.path.join('test', 'metadata.csv'), index=False)
