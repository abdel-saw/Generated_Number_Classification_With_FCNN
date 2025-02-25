import numpy as np
import pandas as pd
from variations import base_digits, generate_variations

def generate_dataset(output_csv="dataset.csv", num_variations=20):
    """
    Generates a dataset of 6x12 binary digits and saves as a CSV file.
    :param output_csv: Name of the CSV file
    :param num_variations: Variations per digit
    """
    digits = base_digits()
    dataset = []

    for label, base_matrix in digits.items():
        variations = generate_variations(base_matrix, num_variations)
        for matrix in variations:
            flat_features = matrix.flatten().tolist()
            dataset.append(flat_features + [int(label)])  # Add label
    
    # Convert to DataFrame and save
    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Dataset saved to {output_csv} with {len(dataset)} samples.")

if __name__ == "__main__":
    generate_dataset()
