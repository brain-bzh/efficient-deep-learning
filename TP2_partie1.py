from minicifar import minicifar_train, minicifar_test, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(minicifar_train, batch_size=800, sampler=train_sampler)
validloader = DataLoader(minicifar_train, batch_size=800, sampler=valid_sampler)
testloader = DataLoader(minicifar_test, batch_size=800)


"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import matplotlib.pyplot as plt

import torch.quantization

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.03, type=float, help="learning rate")
# parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--nepochs", "-n", default=100, type=int, help="number of epochs")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

loss_train = []
loss_test = []
# n_epochs = 50
n_epochs = args.nepochs

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Model
print("==> Building model..")
mymodel = VGG("VGG11")

mymodel = mymodel.to(device)
if device == "cuda":
    mymodel = torch.nn.DataParallel(mymodel)
    cudnn.benchmark = True

# Load checkpoint.
print("==> Resuming from checkpoint..")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
checkpoint = torch.load("./checkpoint/vgg11_minicifar.pth")
mymodel.load_state_dict(checkpoint["net"])
print(f"best_acc = ", checkpoint["acc"])
print(f"last_epoch = ", checkpoint["epoch"])

mymodel.eval()

#32->16bits
mymodel_half=mymodel.half()  # convert all the model parameters to 16 bits half precision

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mymodel_half.parameters(), lr=args.lr, momentum=0.5, weight_decay=5e-4)
# momentum=0.9, weight_decay=5e-4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def test(epoch):
    global best_acc
    global loss_test
    mymodel_half.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs=inputs.half()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = mymodel_half(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

        acc = 100.0 * correct / total

        print("Saving..")
        state = {
            "net": mymodel_half.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/mymodel_half.pth")
        best_acc = acc

        print(f"test_loss = ", test_loss)
        print(f"accuracy = ", acc)

test(0)


