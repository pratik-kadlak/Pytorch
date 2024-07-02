import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    A simple VGG-style convolutional neural network (CNN) for image classification tasks.

    Attributes:
        conv_block_1 (nn.Sequential): The first convolutional block consisting of two convolutional layers,
                                      ReLU activations, and a max pooling layer.
        conv_block_2 (nn.Sequential): The second convolutional block consisting of two convolutional layers,
                                      ReLU activations, and a max pooling layer.
        fc (nn.Sequential): The fully connected block consisting of a flattening layer and a linear layer.

    Args:
        input_shape (int): Number of input channels (e.g., 3 for RGB images).
        hidden_units (int): Number of filters in the convolutional layers.
        output_shape (int): Number of output units (e.g., number of classes for classification).
    """

    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)
        )

    def forward(self, x):
        """
        Defines the forward pass of the TinyVGG model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_shape, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_shape).
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.fc(x)
