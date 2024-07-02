import torch
from torch import nn
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves the trained model's state_dict to the specified directory.

    Args:
        model (torch.nn.Module): The trained model to save.
        target_dir (str): The directory where the model will be saved.
        model_name (str): The name of the model file.

    Raises:
        AssertionError: If model_name does not end with '.pt' or '.pth'.
    """
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)