import os
import torch
from torch import nn
from torchvision import transforms

# import other scripts
import data_setup, model_builder, engine, utils

# set random seed
torch.manual_seed(42)
torch.mps.manual_seed(42)

# set hyper parameters
EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# set up directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# setting yp device agnostic code
device = "mps" if torch.backends.mps.is_available() else "cpu"

# create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# creating dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=data_transform,
                                                                               batch_size=BATCH_SIZE)

# creating model 
model = model_builder.TinyVGG(input_shape=3, 
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)).to(device)

# seeting up loss function
loss_fn = nn.CrossEntropyLoss()

# setting up optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

model_results = engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, EPOCHS)

# Save the model
utils.save_model(model=model,
           target_dir="models",
           model_name="script_mode_tinyvgg_model.pth")