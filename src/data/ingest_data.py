import os
import zipfile
import kaggle

# Define paths relative to the project root
# Ensure these match where you run the script from (usually project root)
RAW_DATA_PATH = os.path.join("data", "raw")
DATASET = "mlg-ulb/creditcardfraud"
ZIP_FILE = os.path.join(RAW_DATA_PATH, "creditcardfraud.zip")
OUTPUT_FILE = os.path.join(RAW_DATA_PATH, "creditcard.csv")

def download_dataset():
    # 1. Create directory if it doesn't exist
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    # 2. Download from Kaggle
    print("Downloading dataset from Kaggle...")
    # This downloads creditcardfraud.zip into data/raw/
    kaggle.api.dataset_download_files(DATASET, path=RAW_DATA_PATH, unzip=False)
    print("Download complete.")

    # 3. Extract the file
    print(f"Extracting {ZIP_FILE} ...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
        zf.extractall(RAW_DATA_PATH)
    
    # 4. Verify Output
    if not os.path.exists(OUTPUT_FILE):
        raise FileNotFoundError(f"Extraction failed. {OUTPUT_FILE} not found.")
    
    print(f"Success! Data available at {OUTPUT_FILE}")

if __name__ == "__main__":
    download_dataset()