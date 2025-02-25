import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import FCNN

# Load dataset
df = pd.read_csv("../data/dataset.csv", header=None)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define model, loss, and optimizer
model = FCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
history = {"loss": [], "accuracy": []}

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y).sum().item() / len(y)
    
    history["loss"].append(loss.item())
    history["accuracy"].append(accuracy)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Save model & training history
torch.save(model.state_dict(), "fcnn_model.pth")
torch.save(history, "training_history.pth")

print("Training complete. Model & history saved.")
