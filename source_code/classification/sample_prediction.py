import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import FCNN

# Load dataset
df = pd.read_csv("../data/dataset.csv", header=None)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)

# Load trained model
model = FCNN()
model.load_state_dict(torch.load("fcnn_model.pth"))
model.eval()

# Select random samples
indices = np.random.choice(len(X), 5, replace=False)
X_samples, y_true = X[indices], y[indices]

# Predict
with torch.no_grad():
    y_pred = torch.argmax(model(X_samples), axis=1).numpy()

# Plot results
plt.figure(figsize=(10, 5))
for i, (sample, true_label, pred_label) in enumerate(zip(X_samples, y_true, y_pred)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(sample.numpy().reshape(6, 12), cmap="gray", aspect="auto")
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
