import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os

# Ajouter le chemin du dossier parent pour importer resnet
sys.path.append(os.path.abspath("../LAB1"))
from resnet import ResNet18

from utils_prune import apply_global_pruning, apply_structured_pruning, apply_thinet_pruning, remove_pruning

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

def evaluate_model(model, testloader, device):
    """Évalue le modèle sur le test set CIFAR10."""
    model.eval()
    correct = 0
    total = 0
    
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f'Accuracy après pruning: {accuracy:.2f}%')

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pruning_method', type=str, default='global', choices=['global', 'structured', 'thinet'])
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloaders(args.batch_size)
    
    # Charger le modèle existant
    model = ResNet18().to(device)
    model.load_state_dict(torch.load("../LAB1/test.pth", map_location=device))
    print("Modèle chargé en half precision !")
    
    # Appliquer le pruning choisi
    if args.pruning_method == 'global':
        apply_global_pruning(model, amount=args.pruning_ratio)
    elif args.pruning_method == 'structured':
        apply_structured_pruning(model, amount=args.pruning_ratio)
    elif args.pruning_method == 'thinet':
        apply_thinet_pruning(model, amount=args.pruning_ratio)
    
    print(f"{args.pruning_method.capitalize()} pruning appliqué avec ratio {args.pruning_ratio}")
    
    # Évaluer le modèle après pruning
    evaluate_model(model, testloader, device)
    
    # Phase d'entraînement après pruning
    train_model(model, trainloader, testloader, device, epochs=10)
    
    
    
if __name__ == "__main__":
    main()