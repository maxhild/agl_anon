import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class OCRDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_files = os.listdir(directory)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        # Assuming the label is the prefix of the filename before an underscore
        # For example, in "label_001.jpg", the label is "label"
        label = self.image_files[idx].split('_')[0]

        return image, label
