import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df):
    """
    Plots the class distribution of a dataset.
    :param df: Pandas DataFrame containing the dataset
    """
    label_counts = df.iloc[:, -1].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="magma")
    plt.xlabel("Digit Label")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=0)
    plt.show()

def load_and_describe(csv_file):
    """
    Loads dataset and prints basic statistics.
    :param csv_file: Path to dataset CSV file
    :return: DataFrame
    """
    df = pd.read_csv(csv_file, header=None)
    print("Dataset Overview:")
    print(df.describe())
    return df
