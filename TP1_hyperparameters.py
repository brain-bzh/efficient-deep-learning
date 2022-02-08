from minicifar import minicifar_train, minicifar_test, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(minicifar_train, batch_size=200, sampler=train_sampler)
validloader = DataLoader(minicifar_train, batch_size=200, sampler=valid_sampler)
testloader = DataLoader(minicifar_test, batch_size=200)


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

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
# parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
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

# trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# Model
print("==> Building model..")
net = VGG("VGG11")

net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]


# Training
def train(epoch):
    global loss_train
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    loss_train.append(train_loss)


def test(epoch):
    global best_acc
    global loss_test
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
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

        loss_test.append(test_loss)

    # Save checkpoint.
    acc = 100.0 * correct / total
    # if acc > best_acc:
    #     print("Saving..")
    #     state = {
    #         "net": net.state_dict(),
    #         "acc": acc,
    #         "epoch": epoch,
    #     }
    #     if not os.path.isdir("checkpoint"):
    #         os.mkdir("checkpoint")
    #     torch.save(state, "./checkpoint/ckpt.pth")
    #     best_acc = acc


def init_weights(layer):
    # If convolutional layer, initialize with normal distribution
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # If fully connected layer, initialize with uniform distribution with bias=0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)


fig1 = plt.figure()
# for lr in [0.3, 0.1, 0.03, 0.01]:
#     net.apply(init_weights)

#     loss_train = []
#     loss_test = []

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.5, weight_decay=1e-3)
#     #! momentum=0.9, weight_decay=5e-4
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#     for epoch in range(start_epoch, start_epoch + n_epochs):
#         train(epoch)
#         test(epoch)
#         scheduler.step()

#     plt.plot(range(n_epochs), loss_train, label=f"Train_{lr}")
#     plt.plot(range(n_epochs), loss_test, label=f"Validation_{lr}")

for wd in [5e-3, 1e-3, 5e-4, 1e-4]:
    net.apply(init_weights)

    loss_train = []
    loss_test = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5, weight_decay=wd)
    #! momentum=0.9, weight_decay=5e-4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + n_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()

    plt.plot(range(n_epochs), loss_train, label=f"Train_wd={wd}")
    plt.plot(range(n_epochs), loss_test, label=f"Validation_wd={wd}")

plt.legend()  # prop={"size": 10}
plt.title("Loss Function", size=10)
plt.xlabel("Epoch", size=10)
plt.ylabel("Loss", size=10)
plt.ylim(ymax=100)
plt.show()
fig1.tight_layout()
fig1.savefig("TP1_report/figure1.png")
