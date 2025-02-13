import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from resnet import ResNet18

def get_dataloaders(batch_size):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def train(model, trainloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    wandb.log({"train_loss": running_loss / len(trainloader), "train_accuracy": acc, "epoch": epoch})
    return running_loss / len(trainloader), acc

def test(model, testloader, criterion, device, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    wandb.log({"test_loss": test_loss / len(testloader), "test_accuracy": acc, "epoch": epoch})
    return test_loss / len(testloader), acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_scheduler', action='store_true')
    args = parser.parse_args()
    
    wandb.init(project="resnet-cifar10", config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloaders(args.batch_size)
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min') if args.use_scheduler else None
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device, epoch)
        test_loss, test_acc = test(model, testloader, criterion, device, epoch)
        if scheduler:
            scheduler.step(test_loss)
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
    
if __name__ == "__main__":
    main()
