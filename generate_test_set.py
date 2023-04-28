import os
import json
import math
import shutil
import requests
import pandas as pd

if not os.path.exists('test_Crystals'):
    os.makedirs('test_Crystals')

def fetch_assets(api_key, contract_address, chain="ethereum", page_size=50):
    all_assets = []

    # Fetch the first page only
    url = f"https://api.nftport.xyz/v0/nfts/{contract_address}?chain={chain}&page_number=1&page_size={page_size}&include=metadata&refresh_metadata=false"

    headers = {
        "accept": "application/json",
        "Authorization": api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        all_assets.extend(data['nfts'])

        with open(os.path.join('test', 'jsons', f'page_1.json'), 'w') as outfile:
            json.dump(data, outfile)

        print(f"Fetched one page of assets")

    else:
        print(f"Error fetching assets: {response.status_code}")

    return all_assets

def download_images(nfts):
    for nft in nfts:
        # Download and save image
        image_url = nft['cached_file_url']
        response = requests.get(image_url, stream=True)
        with open(os.path.join('test_Crystals', filename), 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
            
if __name__ == '__main__':
    contract_address = "0x7AfEdA4c714e1C0A2a1248332c100924506aC8e6" # fvckcrystals contract address
    api_key = "ac5d347e-c8fb-4ab6-9455-2c6823776e3a"
    
    assets = fetch_assets(api_key, contract_address)
    download_images(assets)
    