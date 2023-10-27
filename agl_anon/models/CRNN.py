import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CRNN, self).__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        h, w = input_size
        h //= 4  # Two max pooling layers
        w //= 4  # Two max pooling layers
        
        # Recurrent layers
        self.rnn = nn.LSTM(128 * h, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Dense layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # bidirectional => *2

    def forward(self, x):
        x = self.conv(x)
        # Reshape output from conv layers for input to LSTM
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# Parameters
input_size = (32, 128)  # Example height x width of image after preprocessing
hidden_size = 128
num_layers = 2
num_classes = 37  # Example: 26 alphabets + 10 digits + 1 for CTC 'blank' character
learning_rate = 0.001

# Instantiate the model
model = CRNN(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.CTCLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Example Training loop
# for epoch in range(num_epochs):
#     for batch_images, batch_labels in train_dataloader:
#         # Forward pass, compute loss, backpropagation, optimization
