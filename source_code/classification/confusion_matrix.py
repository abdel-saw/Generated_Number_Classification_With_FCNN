import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

# Predict
with torch.no_grad():
    y_pred = model(X)
    y_pred = torch.argmax(y_pred, axis=1).numpy()

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
