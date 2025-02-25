import pandas as pd
import numpy as np

def check_corrupted_samples(csv_file):
    """
    Detects missing values and invalid entries in the dataset.
    :param csv_file: Path to dataset CSV file
    """
    df = pd.read_csv(csv_file, header=None)
    
    # Check for NaN values
    num_missing = df.isnull().sum().sum()

    # Check for invalid values (anything other than 0 or 1 in feature columns)
    feature_columns = df.iloc[:, :-1]  # Exclude label column
    num_invalid = (feature_columns != 0).astype(int) + (feature_columns != 1).astype(int) - 1
    num_invalid = num_invalid.sum().sum()

    print(f"Missing Values: {num_missing}")
    print(f"Invalid Entries (not 0/1): {num_invalid}")

    if num_missing > 0:
        print("Warning: Dataset contains missing values!")
    
    if num_invalid > 0:
        print("Warning: Dataset contains invalid entries!")

if __name__ == "__main__":
    check_corrupted_samples("../data/dataset.csv")
