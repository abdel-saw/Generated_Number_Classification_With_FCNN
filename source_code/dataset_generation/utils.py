import pandas as pd
import numpy as np

def load_dataset(csv_file):
    """
    Loads the dataset from a CSV file.
    :param csv_file: Path to CSV
    :return: Feature matrix (X), labels (y)
    """
    df = pd.read_csv(csv_file, header=None)
    X = df.iloc[:, :-1].values  # Features (first 72 columns)
    y = df.iloc[:, -1].values   # Labels (last column)
    return X, y

def reshape_digit(sample):
    """
    Reshapes a flattened digit (72 features) into a 6x12 matrix.
    :param sample: 1D array of 72 elements
    :return: 6x12 numpy array
    """
    return np.array(sample).reshape(6, 12)
