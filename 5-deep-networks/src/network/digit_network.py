# src/network/digit_network.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Defines the DigitNetwork CNN architecture for MNIST digit
#              recognition. Architecture follows the project specification:
#              conv -> pool/relu -> conv -> dropout -> pool/relu -> fc -> fc.

import torch.nn as nn
import torch.nn.functional as F


class DigitNetwork(nn.Module):
    """
    Convolutional Neural Network for MNIST digit recognition.

    Architecture (per project spec):
        1. Conv2d:   10 filters, 5x5 kernel
        2. MaxPool2d: 2x2 window  +  ReLU
        3. Conv2d:   20 filters, 5x5 kernel
        4. Dropout:  0.5 drop rate
        5. MaxPool2d: 2x2 window  +  ReLU
        6. Flatten  -> Linear(320 -> 50)  +  ReLU
        7. Linear(50 -> 10)  +  log_softmax

    Input:  (N, 1, 28, 28)  — grayscale MNIST images
    Output: (N, 10)         — log-probabilities over 10 digit classes

    Author: Krushna Sanjay Sharma
    """

    def __init__(self):
        """
        Initialises all layers of the network.
        Layer names match the project specification so that external code
        (e.g. filter analysis) can reference them by name (e.g. self.conv1).
        """
        super(DigitNetwork, self).__init__()

        # First convolution: 1 input channel, 10 output filters, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)

        # Second convolution: 10 input channels, 20 output filters, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)

        # Dropout layer with 50% drop rate (applied after conv2, before pool2)
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer 1: flattened feature map (320) -> 50 nodes
        # 320 = 20 filters * 4 * 4  (after two 2x2 max-pools on 28x28 input)
        self.fc1 = nn.Linear(in_features=320, out_features=50)

        # Fully connected layer 2: 50 nodes -> 10 output classes
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        """
        Computes a forward pass through the network.

        Processing pipeline:
            conv1 -> maxpool -> relu
            conv2 -> dropout -> maxpool -> relu
            flatten -> fc1 -> relu
            fc2 -> log_softmax

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, 28, 28).

        Returns:
            torch.Tensor: Log-probabilities of shape (N, 10).
        """
        # --- Block 1: conv1 -> 2x2 max-pool -> ReLU ---
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))

        # --- Block 2: conv2 -> dropout -> 2x2 max-pool -> ReLU ---
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), kernel_size=2))

        # --- Flatten: (N, 20, 4, 4) -> (N, 320) ---
        x = x.view(-1, 320)

        # --- FC1: 320 -> 50 with ReLU ---
        x = F.relu(self.fc1(x))

        # --- FC2: 50 -> 10 with log_softmax (dim=1 over class axis) ---
        x = F.log_softmax(self.fc2(x), dim=1)

        return x