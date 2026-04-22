# ===================== 1. IMPORTS =====================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import numpy as np

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TRANSFORM 
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


def get_subset(dataset, percentage):
    size = int(len(dataset) * percentage)
    indices = np.random.choice(len(dataset), size, replace=False)
    return Subset(dataset, indices)


#  MODEL
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# FINE-TUNING STRATEGIES

def freeze_all_except_fc(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def unfreeze_last_block(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


# TRAIN FUNCTION 
def train(model, train_loader, val_loader, name, epochs=5):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    train_loss_list = []
    val_acc_list = []
    best_acc = 0

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss_list.append(total_loss)

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                _, pred = torch.max(out, 1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        val_acc_list.append(val_acc)

        print(f"{name} | Epoch {epoch+1} | Loss {total_loss:.3f} | Val Acc {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{name}_best.pth")

    return train_loss_list, val_acc_list, best_acc


def run_experiment(dataset, tag):

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    results = {}
    plots = {}

    # FC Only
    model1 = get_model(10)
    model1 = freeze_all_except_fc(model1)
    loss1, acc1, best1 = train(model1, train_loader, val_loader, f"{tag}_fc")
    results["FC Only"] = best1
    plots["fc"] = (loss1, acc1)

    # Last Block
    model2 = get_model(10)
    model2 = unfreeze_last_block(model2)
    loss2, acc2, best2 = train(model2, train_loader, val_loader, f"{tag}_last")
    results["Last Block"] = best2
    plots["last"] = (loss2, acc2)

    # Full FT
    model3 = get_model(10)
    model3 = unfreeze_all(model3)
    loss3, acc3, best3 = train(model3, train_loader, val_loader, f"{tag}_full")
    results["Full FT"] = best3
    plots["full"] = (loss3, acc3)

    return results, plots


# CREATE DATA SPLITS 
dataset_20 = get_subset(full_dataset, 0.2)
dataset_50 = get_subset(full_dataset, 0.5)
dataset_100 = full_dataset


#  RUN ALL
results_20, plots_20 = run_experiment(dataset_20, "20")
results_50, plots_50 = run_experiment(dataset_50, "50")
results_100, plots_100 = run_experiment(dataset_100, "100")


# SAVE RESULTS 
all_results = {
    "20%": results_20,
    "50%": results_50,
    "100%": results_100
}

with open("results.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("\nFinal Results:")
for k, v in all_results.items():
    print(f"{k}: {v}")

def plot_results(plots, title):

    plt.figure()
    for key in plots:
        plt.plot(plots[key][0], label=f"{key} Loss")
    plt.title(f"{title} Loss")
    plt.legend()
    plt.savefig(f"{title.replace('%', '').replace(' ', '_')}_loss.png")
    plt.close()

    plt.figure()
    for key in plots:
        plt.plot(plots[key][1], label=f"{key} Acc")
    plt.title(f"{title} Accuracy")
    plt.legend()
    plt.savefig(f"{title.replace('%', '').replace(' ', '_')}_acc.png")
    plt.close()

plot_results(plots_20, "20% Data")
plot_results(plots_50, "50% Data")
plot_results(plots_100, "100% Data")