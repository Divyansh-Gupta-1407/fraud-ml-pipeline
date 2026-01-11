import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_PATH = "data/processed/"

# Define output paths for Train and Test
TRAIN_DATA_FILE = os.path.join(PROCESSED_PATH, "train_processed.csv")
TEST_DATA_FILE = os.path.join(PROCESSED_PATH, "test_processed.csv")
SCALER_FILE = os.path.join(PROCESSED_PATH, "scaler.pkl")

def process_data():
    # 1. Load data
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 2. Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 3. SPLIT FIRST (Critical Step)
    # Stratify ensures the test set has the same proportion of fraud as the original
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale Data
    # Fit ONLY on Training data to avoid leakage
    scaler = StandardScaler()
    
    # We only scale 'Time' and 'Amount' as per your original logic
    cols_to_scale = ['Time', 'Amount']
    
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # 5. Apply SMOTE ONLY to Training Data
    print(f"Original Train Count: {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE Train Count: {y_train_resampled.value_counts().to_dict()}")

    # 6. Save Data
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # Save Train (Balanced)
    train_df = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    train_df.to_csv(TRAIN_DATA_FILE, index=False)

    # Save Test (Imbalanced/Real)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(TEST_DATA_FILE, index=False)
    
    # Save scaler for later use (inference)
    joblib.dump(scaler, SCALER_FILE)

    print(f"Data processed.\nTrain saved to: {TRAIN_DATA_FILE}\nTest saved to: {TEST_DATA_FILE}")

if __name__ == "__main__":
    process_data()