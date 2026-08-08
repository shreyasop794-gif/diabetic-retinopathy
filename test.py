from torch.utils.data import DataLoader
from model import DRModel
from data_loader import DRDataset, eval_transform
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate(batch_size: int = 32):
    test_dataset = DRDataset(os.path.join(BASE_DIR, 'test/'), os.path.join(BASE_DIR, 'test_labels.csv'), transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    checkpoint = torch.load(os.path.join(BASE_DIR, 'models/dr_resnet50.pth'), map_location=device)

    # Support both plain state_dict and checkpoint dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model = DRModel(num_classes=5).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = 100.0 * (all_preds == all_labels).mean()
    print(f'Test Accuracy: {acc:.2f}%')

    print("\nClassification report (DR grade 0–4):")
    print(classification_report(all_labels, all_preds, digits=3))

    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()
