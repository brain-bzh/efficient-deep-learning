import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import LAB3.binaryconnect as binaryconnect

import matplotlib.pyplot as plt 
import numpy as np
import sys
from binaryconnect import BC

sys.path.append("/homes/x22weng/efficient-deep-learning-lea/LAB/LAB1/")

from resnet import ResNet18

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network.
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 are already downloaded in the following folder
rootdir = '/opt/img/effdl-cifar10/'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=64,shuffle=True)
testloader = DataLoader(c10test,batch_size=64)

# Fetch the model
loaded_cpt = torch.load('/homes/x22weng/efficient-deep-learning-lea/LAB/LAB3/models/test.pth')
model = ResNet18()
model.load_state_dict(loaded_cpt['model_state_dict'])

# Move model to GPU (if available) and convert to FP16 (from FP32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_half(model):
    model.to(device).half()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for inference
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device).half(), labels.to(device)  # Convert inputs to FP16, keep labels in int format

            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get predicted class index

            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Compute accuracy
    accuracy = 100 * (correct / total)
    print(f"Accuracy: {accuracy:.2f}%")


def binnary_connect():
    # ---- Training loop with binary connect ----
    mymodelbc = binaryconnect.BC(model) ### use this to prepare your model for binarization
    mymodelbc.model = mymodelbc.model.to(device) # it has to be set for GPU training
    ### During training (check the algorithm in the course and in the paper to see the exact sequence of operations)


    mymodelbc.binarization() ## This binarizes all weights in the model
    mymodelbc.restore() ###  This reloads the full precision weights
    ### After backprop
    mymodelbc.clip() ## Clip the weights