import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from model import DRModel
from data_loader import DRDataset, train_transform, eval_transform


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train(batch_size: int = 16, num_epochs: int = 25, lr: float = 1e-4, val_split: float = 0.15):
    # Full training dataset
    full_dataset = DRDataset(os.path.join(BASE_DIR, 'train/'), os.path.join(BASE_DIR, 'train_labels.csv'), transform=train_transform)

    # Create train/validation split for better monitoring
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Validation should not use augmentation
    val_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = DRModel(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_acc = 100.0 * total_correct / total_samples

        # ---------- Validation ----------
        model.eval()
        val_correct = 0
        val_samples = 0
        val_loss_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

                val_loss_total += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

        val_loss = val_loss_total / val_samples
        val_acc = 100.0 * val_correct / val_samples

        print(
            f'Epoch {epoch + 1}/{num_epochs} '
            f'- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% '
            f'- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'val_acc': best_val_acc,
                    'epoch': epoch + 1,
                },
                os.path.join(BASE_DIR, 'models/dr_resnet50.pth'),
            )
            print(f'✅ Saved new best model with Val Acc: {best_val_acc:.2f}%')


if __name__ == "__main__":
    train()
