def preprocess_dataset(dataset_path, height, width, batch_size=32):
    """
    Preprocesses the dataset by converting images to grayscale, resizing them, and 
    normalizing them to range [-1, 1].
    
    Parameters:
        dataset_path (str): Path to the dataset directory.
        height (int): Desired height for resizing images.
        width (int): Desired width for resizing images.
        batch_size (int): Batch size for the DataLoader. Default is 32.
    
    Returns:
        DataLoader: A DataLoader object with preprocessed images.
    """
    
    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.Resize((height, width)),           # Resize to the desired size
        transforms.ToTensor(),                        # Convert image to a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,)),         # Normalize to range [-1, 1] for white on black text
    ])

    # Load the custom dataset with the transformations
    dataset = OCRDataset(directory=dataset_path, transform=transform)
    
    # Create and return a DataLoader to handle batching
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Usage:
dataset_path = 'path_to_your_dataset_directory'
dataloader = preprocess_dataset(dataset_path, height=desired_height, width=desired_width)
