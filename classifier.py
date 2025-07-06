import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
import numpy as np


class ToyDataset(Dataset):
    """Create a dataset from features and labels. Inherits from Pytorch's Dataset class."""

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]
    

class NeuralNetwork(torch.nn.Module):
    """A simple feedforward neural network with one hidden layer."""

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 10),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(10, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def sample_csv(file_path):
    """Load a CSV file, shuffle it, and split it into training and testing sets."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    df = pd.read_csv(file_path)

    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    train_df = df_shuffled.iloc[:125, :]
    test_df = df_shuffled.iloc[125:, :]

    return train_df, test_df


def prepare_dataset(train_df, test_df, batch_size=8):
    """Load the dataset into tensors and create DataLoader objects for training and testing."""

    X_train = torch.tensor(np.array(train_df.iloc[:, 2:]), dtype=torch.float32)
    y_train = torch.tensor(np.array(train_df.iloc[:, :1]), dtype=torch.long).squeeze(1)  # Squeeze to remove extra dimension

    X_test = torch.tensor(np.array(test_df.iloc[:, 2:]), dtype=torch.float32)
    y_test = torch.tensor(np.array(test_df.iloc[:, :1]), dtype=torch.long).squeeze(1)  # Squeeze to remove extra dimension

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,  # NEW: False because of DistributedSampler below
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, test_loader


def compute_accuracy(model, dataloader, device):
    """Compute the accuracy of the model on the given dataloader."""

    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()



if __name__ == "__main__":
    # hyperparameters
    device = "cpu"
    world_size = 1
    num_epochs = 50
    learning_rate = 0.05
    batch_size = 8

    # split the dataset into training and testing sets
    train_df, test_df = sample_csv("data/IRIS.csv")

    # load dataset into tensors
    train_loader, test_loader = prepare_dataset(train_df, test_df, batch_size)
    model = NeuralNetwork(num_inputs=4, num_outputs=3)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # training loop
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:

            features, labels = features.to(device), labels.to(device)  # New: use rank
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            print(f"[GPU{device}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")
            
    model.eval()

    train_acc = compute_accuracy(model, train_loader, device=device)
    print(f"[GPU{device}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f"[GPU{device}] Test accuracy", test_acc)