import pandas as pd

def check_duplicates(csv_file):
    """
    Checks for duplicate samples in the dataset.
    :param csv_file: Path to dataset CSV file
    """
    df = pd.read_csv(csv_file, header=None)
    
    num_samples = len(df)
    num_duplicates = df.duplicated().sum()

    print(f"Total Samples: {num_samples}")
    print(f"Duplicate Samples: {num_duplicates}")

    if num_duplicates > 0:
        print("Warning: Dataset contains duplicate samples!")
    else:
        print("No duplicates found.")

if __name__ == "__main__":
    check_duplicates("../data/dataset.csv")  # Adjust the path as needed
