import kagglehub
import shutil
import os

# Define your local Kaggle directory
local_kaggle_dir = "kaggle"
os.makedirs(local_kaggle_dir, exist_ok=True)

def download_and_save(dataset_id):
    # Download dataset
    cache_path = kagglehub.dataset_download(dataset_id)
    print(f"Downloaded {dataset_id} to cache: {cache_path}")

    # Create subfolder for this dataset
    dataset_name = dataset_id.split("/")[-1]
    target_path = os.path.join(local_kaggle_dir, dataset_name)
    os.makedirs(target_path, exist_ok=True)

    # Move files to local Kaggle dir
    for item in os.listdir(cache_path):
        shutil.move(os.path.join(cache_path, item), target_path)

    print(f"Saved to: {target_path}")

# Download both datasets
# download_and_save("mldtype/sketch-2-image-dataset")
# download_and_save("norod78/sketch2pokemon")
download_and_save("tommasosenatori/flickr8k-sketch")
