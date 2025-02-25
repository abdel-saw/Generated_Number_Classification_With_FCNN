import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_sample_images(csv_file, num_samples=10):
    """
    Randomly selects and plots a few samples per digit.
    :param csv_file: Path to dataset CSV file
    :param num_samples: Number of samples per digit to visualize
    """
    df = pd.read_csv(csv_file, header=None)
    
    # Separate features and labels
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Create a figure for visualization
    fig, axes = plt.subplots(nrows=10, ncols=num_samples, figsize=(num_samples * 1.2, 10))

    for digit in range(10):
        # Get 'num_samples' random samples of this digit
        digit_samples = X[y == digit].sample(n=min(num_samples, len(X[y == digit])), random_state=42).values
        
        for i in range(len(digit_samples)):
            ax = axes[digit, i]
            ax.imshow(digit_samples[i].reshape(6, 12), cmap="gray_r")
            ax.axis("off")

    plt.suptitle("Random Samples per Digit (6x12 Representation)")
    plt.show()

if __name__ == "__main__":
    plot_sample_images("../data/dataset.csv")
