import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import namedtuple
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# === Load FashionMNIST Dataset ===
class_names = [  # Class labels for FashionMNIST (0-9)
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Transform PIL images to tensors normalized to [0, 1]
transform = transforms.ToTensor()

# Download training and test sets if not already present
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# Create data loaders to fetch batches during training/testing
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# === Define a Convolutional Neural Network ===
class CNN(nn.Module):
    def __init__(self, 
                 input_channels=1,
                 conv1_out=6, conv1_kernel=5,
                 conv2_out=16, conv2_kernel=5,
                 use_batchnorm=False,
                 use_dropout=False,
                 dropout_prob=0.5,
                 num_classes=10):
        super().__init__()

        # Build convolutional layers with optional batchnorm
        layers = []
        layers.append(nn.Conv2d(input_channels, conv1_out, conv1_kernel))  # Conv layer: 1 input channel to conv1_out filters
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(conv1_out))  # Normalize outputs of first conv
        layers.append(nn.ReLU())  # Apply non-linearity
        layers.append(nn.MaxPool2d(2, 2))  # Downsample spatial dimensions by 2

        layers.append(nn.Conv2d(conv1_out, conv2_out, conv2_kernel))  # Second conv layer
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(conv2_out))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))

        self.feature_extractor = nn.Sequential(*layers)  # Bundle into one block

        # Compute flattened feature size by passing a dummy input
        example_input = torch.zeros(1, input_channels, 28, 28)  # One dummy image (batch size = 1)
        flattened_size = self.feature_extractor(example_input).numel()  # Total number of output features

        # Build fully connected classifier layers
        fc_layers = [nn.Linear(flattened_size, 120), nn.ReLU()]  # First FC layer + ReLU
        if use_dropout:
            fc_layers.append(nn.Dropout(dropout_prob))  # Randomly deactivate neurons during training
        fc_layers += [nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes)]  # 2nd FC layer + output layer
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.feature_extractor(x)  # Pass through conv layers
        x = x.view(x.size(0), -1)  # Flatten features per image in batch
        x = self.classifier(x)  # Pass through FC layers
        return x

# === Train and Evaluate Baseline Model ===
def train_and_evaluate_baseline(net, separation=1, epoch_num=300, save_path="saved_models/model_Baseline.pth"):
    os.makedirs("saved_models", exist_ok=True)  # Make sure directory exists for saving model
    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # Optimizer with momentum

    epoch_losses = []
    test_accuracies = []
    epochs_to_run = list(range(separation, epoch_num + 1, separation))  # Epochs where we save loss/accuracy

    for epoch in range(1, epoch_num + 1):
        net.train()  # Enable training mode (e.g. dropout on)
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()  # Reset gradients
            outputs = net(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights
            running_loss += loss.item()  # Accumulate batch loss

        if epoch in epochs_to_run:
            avg_loss = running_loss / len(train_dataloader)  # Average over batches
            epoch_losses.append(avg_loss)
            print(f'Epoch {epoch} average loss: {avg_loss:.4f}')

            # Evaluation phase
            correct = 0
            total = 0
            net.eval()
            with torch.no_grad():
                for images, labels in test_dataloader:
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)  # Pick class with highest score
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()  # Count correct predictions
            acc = correct / total
            test_accuracies.append(acc)
            print(f'Epoch {epoch} test accuracy: {100 * acc:.2f}%')

    torch.save(net.state_dict(), save_path)  # Save model weights
    print(f"Baseline model saved to {save_path}")

    with open("baseline_metrics.pkl", "wb") as f:
        pickle.dump({
            "epochs": epochs_to_run,
            "losses": epoch_losses,
            "accuracies": test_accuracies
        }, f)  # Save loss/accuracy logs to disk

    return "Baseline", test_accuracies[59]  # Return label + last recorded accuracy

# === Plot Confusion Matrix ===
def plot_confusion_matrix(model, dataloader, class_names, title="Confusion Matrix (Baseline)", save_path="Confusion_Matrix_Baseline.png"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)  # Compute confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(title, fontsize=20)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# === Train and Evaluate Variant Model ===
def train_and_evaluate(net, name, epochs=60, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for _ in range(epochs):
        net.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate final model
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f'{name} Accuracy: {100 * acc:.2f}%')
    torch.save(net.state_dict(), os.path.join(save_dir, f"model_{name.replace(' ', '_')}.pth"))
    return name, acc

# === Define Model Variants ===
ModelVariant = namedtuple("ModelVariant", ["name", "params"])
models_to_run = [
    ModelVariant("Wider Layers", { "conv1_out": 8, "conv2_out": 32 }),
    ModelVariant("Varied Kernels", { "conv1_kernel": 3, "conv2_kernel": 7 }),
    ModelVariant("With Dropout", { "use_dropout": True, "dropout_prob": 0.3 }),
    ModelVariant("With BatchNorm", { "use_batchnorm": True }),
]

# === Execute Training and Evaluation ===
results = []

print("\nTraining Baseline Model...")
baseline_model = CNN()
results.append(train_and_evaluate_baseline(baseline_model))
plot_confusion_matrix(baseline_model, test_dataloader, class_names)  # Visualize prediction breakdown

for variant in models_to_run:
    print(f"\nTesting model definition: {variant.name}")
    try:
        model = CNN(**variant.params)
        print(f"Model {variant.name} defined successfully.")
        print(f"Training model: {variant.name}")
        results.append(train_and_evaluate(model, variant.name))
    except Exception as e:
        print(f"Error in model '{variant.name}': {e}")

# === Display and Sort Results ===
df_results = pd.DataFrame(results, columns=["Model", "Test Accuracy"])
df_results["Test Accuracy (%)"] = df_results["Test Accuracy"] * 100

df_results = df_results.drop("Test Accuracy", axis=1)
df_results = df_results.sort_values("Test Accuracy (%)", ascending=False).reset_index(drop=True)

print("\n=== Model Comparison ===")
print(df_results.to_string(index=False))

# === Print Model Import Statements ===
print("\n=== Import Statements ===")
for name in df_results["Model"]:
    print(f"from saved_models import model_{name.replace(' ', '_')}")
