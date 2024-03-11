import torch
import torch.nn as nn
import torch.nn.functional as F

class stDNN(nn.Module):
    def __init__(self, num_channels=246, num_classes=2):
        super(stDNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=256, kernel_size=7)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2)
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        # First 1D CNN block
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Second 1D CNN block
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # Temporal averaging
        x = torch.mean(x, 2)
        # Classification layer
        x = torch.sigmoid(self.fc(x))
        return x
