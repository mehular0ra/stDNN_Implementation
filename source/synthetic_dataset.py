from torch.utils.data import Dataset, DataLoader
import torch

class fMRIDataset(Dataset):
    def __init__(self, x, y):
        """
        Args:
            x (Tensor): Input features with shape (num_samples, num_channels, sequence_length).
            y (Tensor): Labels with shape (num_samples,).
        """
        assert x.size(0) == y.size(0), "The number of samples in x and y should be equal."
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.
        Returns:
            tuple: (feature, label) where feature is the input data at index `idx` and label is its corresponding label.
        """
        return self.x[idx], self.y[idx]


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

# Create a DataLoader
batch_size = 32
dataloader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)

# Example of iterating over the DataLoader
for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    # Here, you can feed 'features' and 'labels' to your model for training
    break  # Breaking here just to demonstrate; remove this in real training loops