import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from model import FCNN

# Load dataset
df = pd.read_csv("../data/dataset.csv", header=None)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values  # Features and labels

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)

# Load trained model
model = FCNN()
model.load_state_dict(torch.load("../models/fcnn_model.pth"))
model.eval()

# Predict on dataset
with torch.no_grad():
    y_pred = model(X)
    y_pred = torch.argmax(y_pred, axis=1).numpy()

# Classification Report
print("🔍 Classification Report:")
print(classification_report(y, y_pred))
