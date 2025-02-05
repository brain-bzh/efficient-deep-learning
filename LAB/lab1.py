import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18  # Change this to other models like vgg16, densenet121

# Set seed for reproducibility
seed = 2147483647
torch.manual_seed(seed)
np.random.seed(seed)

# Data preprocessing
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

# Load CIFAR-10 dataset
rootdir = '/opt/img/effdl-cifar10/'
trainset = datasets.CIFAR10(root=rootdir, train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root=rootdir, train=False, download=True, transform=transform_test)

# Subset for faster training
num_samples_subset = 15000
indices = np.random.choice(len(trainset), num_samples_subset, replace=False)
trainset_subset = Subset(trainset, indices)

trainloader = DataLoader(trainset_subset, batch_size=32, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

###############################################################################################################
# Define the model
model = resnet18(num_classes=10)  # Change this line to try different models
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs=10
device="cuda" if torch.cuda.is_available() else "cpu"

# Training function
def train_model(model, trainloader, testloader, epochs=10):
    model.to(device)
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

            # Training accuracy
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * correct_train / total_train

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

        print(f"Epoch {epoch+1}: "
              f"Train Loss: {running_loss/len(trainloader):.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Test Acc: {test_accuracy:.2f}%")

    print("Training complete!")

# Train the model
train_model(model, trainloader, testloader, epochs)

# Save the model
state = {
    'net': model.state_dict(),
    'hyperparam': 0.001  # Learning rate
}
torch.save(state, 'cifar10_model.pth')
print("Model saved!")

# Load the model
loaded_cpt = torch.load('cifar10_model.pth')
hparam_bestvalue = loaded_cpt['hyperparam']
model.load_state_dict(loaded_cpt['net'])
model.eval()
print("Model loaded!")

# Plot accuracy vs. number of parameters
models = {"ResNet18": resnet18(num_classes=10)}
num_params = {}
accuracy = {}

for name, mdl in models.items():
    num_params[name] = sum(p.numel() for p in mdl.parameters())
    accuracy[name] = np.random.uniform(80, 95)  # Placeholder for real accuracy from experiments

plt.figure()
plt.scatter(num_params.values(), accuracy.values(), marker='o')
for name in num_params:
    plt.text(num_params[name], accuracy[name], name)
plt.xlabel("Number of Parameters")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy vs. Number of Parameters")
plt.grid()
plt.show()
