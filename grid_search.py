import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Hyperparameters
# batch_size = 64
# learning_rate = 0.1
# num_epochs = 10
# weight_decay = 1e-4

# Normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download CIFAR-10 dataset and apply transformations
train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR-10', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR-10', train=False, download=True, transform=transform)


def write_record_to_json(record):
    file_path = "grid_search_hyperparameters.json"
    try:
        # Load existing data from the file
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file does not exist, initialize data as an empty list
        data = []

    # Append the new record to the data
    data.append(record)

    # Save the updated data to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)



def test_model(resnet18_model, test_loader):
    resnet18_model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = resnet18_model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy
def compute_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Update counts
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy

def train_model(batch_size, learning_rate, weight_decay):
    train_accuracies = []
    train_losses = []  
    # Create DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Import ResNet-18 model from torchvision
    resnet18_model = torchvision.models.resnet18(pretrained=False, num_classes=10)  # Set num_classes to 10 for CIFAR-10
    resnet18_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet18_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    # Training loop
    for epoch in range(num_epochs):
        resnet18_model.train()  # Set the model to training mode
        i = 0
        for images, labels in train_loader:
            i+=1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet18_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Print training statistics
            current_lr = optimizer.param_groups[0]['lr']
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}')
        scheduler.step()
        training_accuracy = compute_accuracy(resnet18_model, train_loader, device)
        train_accuracies.append(training_accuracy)

        if ((epoch + 1) % 5 == 0):
            accuracy = test_model(resnet18_model,test_loader)
            results = {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": epoch,
                "weight_decay": weight_decay,
                "accuracy": accuracy,
                "train_losses": train_losses,
                "train_accuracies": train_accuracies
            }
            write_record_to_json(results)
    print(f"The following hyperparameters have been tested: {batch_size},{learning_rate},{weight_decay}")

        
for batch_size in [32,64,128]:
    for learning_rate in [0.1,0.01,0.001,0.0001]:
        for weight_decay in [1e-4,3e-4,5e-4]:
            num_epochs = 25
            train_model(batch_size, learning_rate, weight_decay)


