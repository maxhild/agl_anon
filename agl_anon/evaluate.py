import torch
import argparse
from torch.utils.data import DataLoader
from datasets.ocr_dataset import OCRDataset
from models.crnn import CRNN

# Command-line arguments
parser = argparse.ArgumentParser(description='Evaluate the CRNN model')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained CRNN model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
args = parser.parse_args()

# Configuration
MODEL_PATH = args.model_path
BATCH_SIZE = args.batch_size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model = CRNN()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()  # Set the model to evaluation mode

# Data loading
test_dataset = OCRDataset(train=False)   # Adjust accordingly
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_dataloader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)
        
        # Compute predictions
        _, predicted = outputs.max(1)
        
        # Update correct and total counts (modify based on your use-case)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

# Print results
accuracy = 100 * correct / total
print(f"Accuracy on test data: {accuracy:.2f}%")

# Optional: Visualize some predictions, save results, etc.

