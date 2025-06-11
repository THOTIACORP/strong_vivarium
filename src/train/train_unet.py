import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Transformações conjuntas para imagem e máscara
# Veja todas configurações de albumentations: https://explore.albumentations.ai/
class JointTransform:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])


    def __call__(self, image, mask):
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        image = self.image_transform(image)

        torch.manual_seed(seed)
        mask = self.image_transform(mask)

        return self.normalize(self.to_tensor(image)), torch.as_tensor(np.array(mask), dtype=torch.long)


# Dataset
class ThermalMouseDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]

        image_bases = {os.path.splitext(f)[0]: f for f in image_files}
        mask_bases = {os.path.splitext(f)[0]: f for f in mask_files}
        common_bases = sorted(set(image_bases.keys()) & set(mask_bases.keys()))

        self.images_filenames = [image_bases[b] for b in common_bases]
        self.masks_filenames = [mask_bases[b] for b in common_bases]

        assert len(self.images_filenames) > 0, "Nenhuma imagem com máscara correspondente encontrada!"

    def __len__(self):
        return len(self.images_filenames)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Redimensiona a máscara ANTES da transformação para evitar problemas
        mask = mask.resize((512, 512), resample=Image.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

# UNet modelo
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 512))
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def center_crop(self, enc_feat, target_size):
        _, _, h, w = target_size
        _, _, H, W = enc_feat.size()
        delta_h = (H - h) // 2
        delta_w = (W - w) // 2
        return enc_feat[:, :, delta_h:delta_h+h, delta_w:delta_w+w]

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4 = self.center_crop(enc4, dec4.size())
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.center_crop(enc3, dec3.size())
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.center_crop(enc2, dec2.size())
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.center_crop(enc1, dec1.size())
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.conv_last(dec1)

# Métricas
def evaluate_metrics(outputs, masks, n_classes=4):
    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    preds_np = preds.cpu().numpy().flatten()
    masks_np = masks.cpu().numpy().flatten()

    ious = []
    f1s = []
    accs = []

    for cls in range(1, n_classes):
        pred_cls = preds_np == cls
        true_cls = masks_np == cls

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        iou = intersection / union if union != 0 else 1.0
        f1 = f1_score(true_cls, pred_cls, zero_division=1)
        acc = accuracy_score(true_cls, pred_cls)

        ious.append(iou)
        f1s.append(f1)
        accs.append(acc)

    return {
        "iou_mean": np.mean(ious),
        "f1_mean": np.mean(f1s),
        "pixel_accuracy": (preds_np == masks_np).sum() / len(preds_np),
        "mean_accuracy": np.mean(accs),
        "iou_per_class": ious,
        "f1_per_class": f1s
    }

def train_or_validate(model, dataloader, optimizer, criterion, device, n_classes, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    total_metrics = []

    for i, (images, masks) in enumerate(tqdm(dataloader, desc="Treinando" if training else "Validando")):
        print(f"Batch {i}: images.shape={images.shape}, masks.shape={masks.shape}")
        images, masks = images.to(device), masks.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, masks)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        metrics = evaluate_metrics(outputs, masks, n_classes)
        total_metrics.append(metrics)

    avg_metrics = {
        k: np.mean([m[k] for m in total_metrics]) for k in total_metrics[0]
    }
    return total_loss / len(dataloader), avg_metrics

def save_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.savefig(save_path)
    plt.close()

def save_roc_curve(y_true, y_probs, n_classes, save_path):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Curva ROC por Classe")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()

def save_sankey(y_true, y_pred, save_path):
    labels = [f'Classe {i}' for i in sorted(set(y_true) | set(y_pred))]
    source = []
    target = []
    value = []

    cm = confusion_matrix(y_true, y_pred)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            source.append(i)
            target.append(j + len(cm))  # offset
            value.append(cm[i][j])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels + labels,
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.write_html(save_path)
os.makedirs("metrics", exist_ok=True)

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir = "ratos/novos_ratos"
    masks_dir = "ratos/mascaras"

    transform = JointTransform()


    full_dataset = ThermalMouseDataset(images_dir, masks_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)


    model = UNet(in_channels=3, out_channels=4).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print(f"\n--- Época {epoch+1}/{num_epochs} ---")
        train_loss, train_metrics = train_or_validate(model, train_loader, optimizer, criterion, device, 4, training=True)
        val_loss, val_metrics = train_or_validate(model, val_loader, optimizer, criterion, device, 4, training=False)

        print(f"Treino  - Loss: {train_loss:.4f}, mIoU: {train_metrics['iou_mean']:.4f}, F1: {train_metrics['f1_mean']:.4f}")
        print(f"Validação - Loss: {val_loss:.4f}, mIoU: {val_metrics['iou_mean']:.4f}, F1: {val_metrics['f1_mean']:.4f}")

        history["train"].append({"loss": train_loss, **train_metrics})
        history["val"].append({"loss": val_loss, **val_metrics})

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, f"unet_checkpoint_epoch_{epoch+1}.pth")


    # Após todas as épocas
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            all_probs.extend(torch.softmax(outputs, dim=1).permute(0, 2, 3, 1).reshape(-1, outputs.shape[1]).cpu().numpy())

    # Salvar métricas
    classes = list(range(1, 4))  # ignora classe 0 (fundo)
    save_confusion_matrix(all_targets, all_preds, classes, "metrics/confusion_matrix.png")
    save_roc_curve(np.array(all_targets), np.array(all_probs), n_classes=4, save_path="metrics/roc_curve.png")
    save_sankey(all_targets, all_preds, "metrics/sankey.html")

    # Salvar histórico
    with open("metrics/history.json", "w") as f:
        json.dump(history, f, indent=4)



if __name__ == "__main__":
    main()
