import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchinfo import summary
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from tqdm import tqdm


# ---------- Transformações ----------

class JointTransform:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    def __call__(self, image, mask):
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        image = self.image_transform(image)

        torch.manual_seed(seed)
        mask = self.image_transform(mask)

        return self.normalize(self.to_tensor(image)), torch.as_tensor(np.array(mask), dtype=torch.long)

class ValTransform:
    def __init__(self):
        self.resize = transforms.Resize((512, 512))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    def __call__(self, image, mask):
        image = self.resize(image)
        mask = self.resize(mask)

        image = self.normalize(self.to_tensor(image))
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


# ---------- Dataset ----------

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, use_mixup=False, use_cutmix=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        image_bases = {os.path.splitext(f)[0]: f for f in image_files}
        mask_bases = {os.path.splitext(f)[0]: f for f in mask_files}
        common_bases = sorted(set(image_bases) & set(mask_bases))

        self.images_filenames = [image_bases[b] for b in common_bases]
        self.masks_filenames = [mask_bases[b] for b in common_bases]

        assert self.images_filenames, "Nenhuma imagem com máscara correspondente encontrada!"

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).resize((512, 512), resample=Image.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)

        apply_aug = random.random()
        if self.use_mixup and apply_aug < 0.5:
            return self.apply_mixup(image, mask)
        if self.use_cutmix and apply_aug >= 0.5:
            return self.apply_cutmix(image, mask)

        return image, mask

    def apply_mixup(self, image1, mask1):
        lam = np.random.beta(0.4, 0.4)
        idx2 = random.randint(0, len(self.images_filenames) - 1)
        image2, mask2 = self.__getitem__(idx2)

        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_mask = mask1 if lam > 0.5 else mask2

        return mixed_image, mixed_mask

    def apply_cutmix(self, image1, mask1):
        idx2 = random.randint(0, len(self.images_filenames) - 1)
        image2, mask2 = self.__getitem__(idx2)

        _, h, w = image1.shape
        cut_rat = np.sqrt(1. - np.random.beta(1.0, 1.0))
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, w)
        y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, h)

        image1[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        mask1[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

        return image1, mask1


# ---------- Modelo UNet ----------

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
        dh, dw = (H - h) // 2, (W - w) // 2
        return enc_feat[:, :, dh:dh + h, dw:dw + w]

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.dec4(torch.cat([self.upconv4(bottleneck), self.center_crop(enc4, self.upconv4(bottleneck).size())], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), self.center_crop(enc3, self.upconv3(dec4).size())], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), self.center_crop(enc2, self.upconv2(dec3).size())], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), self.center_crop(enc1, self.upconv1(dec2).size())], dim=1))

        return self.conv_last(dec1)


# ---------- Avaliação ----------

def evaluate_metrics(outputs, masks, n_classes=4):
    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    preds_np = preds.cpu().numpy().flatten()
    masks_np = masks.cpu().numpy().flatten()

    ious, f1s, accs = [], [], []
    for cls in range(1, n_classes):
        pred_cls = preds_np == cls
        true_cls = masks_np == cls
        iou = np.logical_and(pred_cls, true_cls).sum() / np.logical_or(pred_cls, true_cls).sum() if np.logical_or(pred_cls, true_cls).sum() > 0 else 1.0
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
    total_loss, total_metrics = 0.0, []

    for i, (images, masks) in enumerate(tqdm(dataloader, desc="Treinando" if training else "Validando")):
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
        total_metrics.append(evaluate_metrics(outputs, masks, n_classes))

    avg_metrics = {k: np.mean([m[k] for m in total_metrics]) for k in total_metrics[0]}
    return total_loss / len(dataloader), avg_metrics


# ---------- Métricas Visuais ----------

def save_confusion_matrix(y_true, y_pred, classes, save_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred, labels=classes), display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.savefig(save_path)
    plt.close()

def save_roc_curve(y_true, y_probs, n_classes, save_path):
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        plt.plot(fpr, tpr, label=f'Classe {i} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC"); plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_sankey(y_true, y_pred, save_path):
    labels = [f'Classe {i}' for i in sorted(set(y_true + y_pred))]
    cm = confusion_matrix(y_true, y_pred)
    source, target, value = [], [], []
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            source.append(i)
            target.append(j + len(cm))
            value.append(val)
    fig = go.Figure(data=[go.Sankey(node=dict(label=labels + labels), link=dict(source=source, target=target, value=value))])
    fig.write_html(save_path)


# ---------- Treinamento ----------

def main():
    os.makedirs("metrics", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir = "../models/model_unet_freedom/data/images/thermal_images"
    masks_dir = "../models/model_unet_freedom/data/images/output"

    transform_train = JointTransform()
    transform_val = ValTransform()

    dataset = SegmentationDataset(images_dir, masks_dir, transform=transform_train, use_mixup=True, use_cutmix=True)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    val_set.dataset.transform = transform_val
    val_set.dataset.use_mixup = val_set.dataset.use_cutmix = False

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = UNet().to(device)
        
    # Mostra estrutura detalhada
    summary(model, input_size=(1, 3, 512, 512), col_names=["input_size", "output_size", "num_params", "trainable"])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = {"train": [], "val": []}

    for epoch in range(10):
        print(f"\n--- Época {epoch+1}/10 ---")
        train_loss, train_metrics = train_or_validate(model, train_loader, optimizer, criterion, device, 4, True)
        val_loss, val_metrics = train_or_validate(model, val_loader, optimizer, criterion, device, 4, False)
        print(f"Train Loss: {train_loss:.4f}, mIoU: {train_metrics['iou_mean']:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, mIoU: {val_metrics['iou_mean']:.4f}")
        history["train"].append({"loss": train_loss, **train_metrics})
        history["val"].append({"loss": val_loss, **val_metrics})
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}, f"unet_checkpoint_epoch_{epoch+1}.pth")

    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            all_probs.extend(torch.softmax(outputs, dim=1).permute(0, 2, 3, 1).reshape(-1, 4).cpu().numpy())

    save_confusion_matrix(all_targets, all_preds, classes=[1, 2, 3], save_path="metrics/confusion_matrix.png")
    save_roc_curve(np.array(all_targets), np.array(all_probs), n_classes=4, save_path="metrics/roc_curve.png")
    save_sankey(all_targets, all_preds, "metrics/sankey.html")
    with open("metrics/history.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
