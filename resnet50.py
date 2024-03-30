import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import random

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load SVHN dataset (small subset) M22AIE214 LAST DIGIT IS 4
train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

subset_indices = random.sample(range(len(train_dataset)), 1000)
train_subset = torch.utils.data.Subset(train_dataset, subset_indices)

# data loaders
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer for classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 20)


model = model.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizers
optimizers = {
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'Adagrad': optim.Adagrad(model.parameters(), lr=0.01),
    'Adadelta': optim.Adadelta(model.parameters(), lr=1.0),
}

# Training loop
num_epochs = 3
for optimizer_name, optimizer in optimizers.items():
    print(f"Training with {optimizer_name} optimizer...")
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss with {optimizer_name} optimizer')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy with {optimizer_name} optimizer')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluate on test set
    model.eval()
    correct_top5 = 0
    total_top5 = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            total_top5 += labels.size(0)
            for pred, label in zip(predicted_top5, labels):
                if label in pred:
                    correct_top5 += 1

    top5_accuracy = correct_top5 / total_top5
    print(f"Final Top-5 Test Accuracy with {optimizer_name} optimizer: {top5_accuracy:.4f}")
