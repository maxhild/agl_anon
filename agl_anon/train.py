import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.ocr_dataset import OCRDataset
from models.crnn import CRNN
import argparse

# Command-line arguments
parser = argparse.ArgumentParser(description='Train the CRNN model')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
args = parser.parse_args()

# Configuration and hyperparameters
EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data loading
train_dataset = OCRDataset(train=True)   # You'll need to define the appropriate parameters
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = OCRDataset(train=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, and Optimizer
model = CRNN().to(DEVICE)
criterion = nn.CTCLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item()}")

    # Validation loop
    model.eval()
    with torch.no_grad():
        for data, targets in val_dataloader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            loss = criterion(outputs, targets)
            # Compute validation metrics here (e.g., accuracy)

    # Print validation results
    # ...

# Save the model
torch.save(model.state_dict(), 'crnn_model.pth')

print("Training complete!")

