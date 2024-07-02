import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(train_dir: str, 
                       test_dir: str, 
                       transform: transforms.Compose, 
                       batch_size: int):
    """
    Creates PyTorch DataLoader objects for training and testing datasets.

    Args:
        train_dir (str): Directory path for the training dataset.
        test_dir (str): Directory path for the testing dataset.
        transform (transforms.Compose): Transformations to be applied to the images.
        batch_size (int): Number of samples per batch to load.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is NUM_WORKERS.

    Returns:
        tuple: A tuple containing the training DataLoader and testing DataLoader.
            - train_dataloader (DataLoader): DataLoader for the training dataset.
            - test_dataloader (DataLoader): DataLoader for the testing dataset.
            - class_names (List): Categories of the data
    """

    # use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # get class names
    class_names = train_data.classes

    # turn the dataset into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader, class_names
