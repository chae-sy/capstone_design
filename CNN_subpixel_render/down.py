import os
import requests
from tqdm import tqdm
from zipfile import ZipFile

def download_with_progress(url, dest_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    base_dir = ".\data"
    urls = {
        "train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "val":   "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    }
    
    for split, url in urls.items():
        zip_filename = os.path.join(base_dir, f"DIV2K_{split}_HR.zip")
        extract_folder = os.path.join(base_dir, split)
        
        print(f"Processing {split} split:")
        download_with_progress(url, zip_filename)
        extract_zip(zip_filename, extract_folder)
        print(f"Extracted to {extract_folder}")
        os.remove(zip_filename)
        print(f"Removed archive {zip_filename}\n")
