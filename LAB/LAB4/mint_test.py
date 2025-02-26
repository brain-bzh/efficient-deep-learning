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

from utils_prune import apply_global_pruning, remove_pruning

def get_dataloaders(batch_size):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    rootdir = '/opt/img/effdl-cifar10/'
    testset = torchvision.datasets.CIFAR10(rootdir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testloader = get_dataloaders(args.batch_size)
    
    # Charger le modèle existant
    model = ResNet18().to(device)
    model.load_state_dict(torch.load("../LAB1/final_model.pth", map_location=device))
    print("Modèle chargé !")
    
    # Appliquer le pruning
    apply_global_pruning(model, amount=args.pruning_ratio)
    print(f"Pruning global appliqué avec ratio {args.pruning_ratio}")
    
    # Évaluer le modèle après pruning
    evaluate_model(model, testloader, device)
    
    # Optionnel : retirer le pruning pour fixer la structure
    remove_pruning(model)
    print("Pruning retiré !")
    
if __name__ == "__main__":
    main()
