import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sys
from binaryconnect import BC
sys.path.append("/homes/x22weng/efficient-deep-learning-lea/LAB/LAB1/")
from resnet import ResNet18

#############################################################################################################################################################
def binary_train(args):
    # Initialize Weights & Biases
    wandb.init(project="deep-learning-lab3", config=args.__dict__ , name="binary_training")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data preprocessing
    normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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

    c10train = CIFAR10(args.data_path,train=True,download=True,transform=transform_train)
    c10test = CIFAR10(args.data_path,train=False,download=True,transform=transform_test)

    trainloader = DataLoader(c10train,batch_size=args.batch_size,shuffle=True)
    testloader = DataLoader(c10test,batch_size=args.batch_size)

    # Fetch the model
    loaded_cpt = torch.load('/homes/x22weng/efficient-deep-learning-lea/LAB/LAB3/models/test.pth')
    model = ResNet18()
    model.load_state_dict(loaded_cpt['model_state_dict'])

    # Get the binary model
    mymodelbc = BC(model)
    mymodelbc.model = mymodelbc.model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Training loop
    for epoch in range(args.epochs):
        mymodelbc.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # 1️) **Binarize weights before forward pass**
            mymodelbc.binarization()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 2️) **Restore full precision before backpropagation**
            mymodelbc.restore()
            loss.backward()
            optimizer.step()

            # 3️) **Clip weights after update**
            mymodelbc.clip()

            running_loss += loss.item()

            # Training accuracy
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(trainloader)

        print(f"TRAINING: Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Learning rate: {optimizer.param_groups[0]['lr']}")

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

        test_accuracy = 100 * (correct_test / total_test)
        test_loss= test_loss / len(testloader)

        scheduler.step(test_loss / len(testloader))
        
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "train_learning_rate": optimizer.param_groups[0]['lr'],
        })
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_accuracy:.2f}% "
              f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        
    print("Training complete my boss")
    wandb.finish()
    # Save model along with training hyperparameters
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
    }

    torch.save(checkpoint, args.save_path)
    print(f"Model and training details saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for regularization")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_path", type=str, default="/opt/img/effdl-cifar10/", help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default="models/cnn_model.pth", help="Path to save the model")
    


    args = parser.parse_args()
    binary_train(args)

    ### QUESTIONS : 
    # - Binarize before evaluating?
    # - do we perform binary connect on the pretrained CIFAR from the github repo of the researchers or our trained lastly trained model?
    # Is is a new training from scratch? If we use the previous model, how many sample to feed the model ?