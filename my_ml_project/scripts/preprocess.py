import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
CSV_FILEPATH = os.getenv("CSV_FILEPATH")

if not CSV_FILEPATH:
    raise ValueError("CSV_FILEPATH not found in .env file.")

# Load the dataset
print(f"Loading dataset from: {CSV_FILEPATH}")
try:
    data = pd.read_csv(CSV_FILEPATH)
    print(f"Dataset loaded successfully with shape: {data.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Preview dataset
print("Dataset preview:")
print(data.head())

# Step 1: Set target (Disease) and features (Symptoms)
TARGET_COLUMN = data.columns[0]  # First column is the target (Disease)
SYMPTOM_COLUMNS = data.columns[1:]  # All other columns are features

print(f"Target column: {TARGET_COLUMN}")
print(f"Feature columns: {SYMPTOM_COLUMNS}")

# Separate features (X) and target (y)
X = data[SYMPTOM_COLUMNS]
y = data[TARGET_COLUMN]

# Step 2: Handle missing values (if any)
X = X.fillna(0)  # Fill missing values in symptoms with 0 (adjust if needed)

# Step 3: Encode the target variable (Disease)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Encoded target classes: {label_encoder.classes_}")

# Step 4: Scale numeric features (Symptom values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split into train/validation/test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Save processed datasets
output_dir = "data/processed/"
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
pd.DataFrame(X_val).to_csv(f"{output_dir}/X_val.csv", index=False)
pd.DataFrame(y_val).to_csv(f"{output_dir}/y_val.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)

print(f"Processed data saved in {output_dir}")
