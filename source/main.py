import torch

from model_stDNN import stDNN
from synthetic_dataset import fMRIDataset
from train import cross_validate_model


# Create synthetic data
num_samples = 1000
num_channels = 246
num_classes = 2
sequence_length = 1000  # Assuming each sample is a time series with 1000 time points

# Synthetic features
x = torch.randn(num_samples, num_channels, sequence_length)
# Synthetic labels: Generate random binary labels
y = torch.randint(0, num_classes, (num_samples,))

# Instantiate the dataset
synthetic_dataset = fMRIDataset(x, y)

cross_validate_model(stDNN, synthetic_dataset, num_splits=2, batch_size=32, epochs=15, learning_rate=0.0001)

