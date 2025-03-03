import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import sys
import os
import wandb

# Ajouter le chemin du dossier parent pour importer resnet
sys.path.append(os.path.abspath("../LAB1"))
from resnet import ResNet18

def get_dataloaders(batch_size):
    rootdir = '/opt/img/effdl-cifar10/'
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
    trainset = torchvision.datasets.CIFAR10(root=rootdir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=rootdir, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def train_model(model, trainloader, testloader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)
        
        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(trainloader)

        # Evaluate on test set
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_test += predicted.eq(labels).sum().item()
                total_test += labels.size(0)
        
        test_accuracy = 100 * correct_test / total_test
        avg_test_loss = test_loss / len(testloader)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "test_loss": avg_test_loss, "test_accuracy": test_accuracy})

#L1 structured pruning then general pruning then tuning

def apply_l1_structured_pruning(model, amount=0.2):
    """Applies structured pruning to convolutional filters."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)

def apply_global_pruning(model, amount=0.2):
    """Applies global unstructured pruning based on L1 norm."""
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)


def main():
    print("Starting pruning process...")
    # parsers for the two prunings methods (structured and global)
    parser = argparse.ArgumentParser()
    parser.add_argument('--structured',type=float, default=0.2)
    parser.add_argument('--glob',type=float, default=0.2)
    parser.add_argument('--epochs',type=int, default=30)
    parser.add_argument('--batch_size',type=int, default=64)
    args = parser.parse_args()

    wandb.init(project="resnet-pruning", config=vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    trainloader, testloader = get_dataloaders(args.batch_size)
    print("Data loaded !")

    # Charger le modèle existant
    model = ResNet18().to(device)
    checkpoint = torch.load("../LAB1/test.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Modèle chargé en half precision !")

    # Appliquer le structured pruning
    apply_l1_structured_pruning(model, amount=args.structured)
    # Appliquer le global pruning
    apply_global_pruning(model, amount=args.glob)
    # Train the model
    train_model(model, trainloader, testloader, device, epochs=args.epochs)
    # Save the model
    model.half()
    torch.save(model.state_dict(), 'pruned_model.pth')

    wandb.save('pruned_model.pth')
    wandb.finish()

if __name__ == '__main__':
    main()
