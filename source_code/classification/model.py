import torch
import torch.nn as nn
import torch.optim as optim

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(72, 20)  # First hidden layer
        self.fc2 = nn.Linear(20, 10)  # Second hidden layer
        self.fc3 = nn.Linear(10, 10)  # Output layer (10 classes)
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax for classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
