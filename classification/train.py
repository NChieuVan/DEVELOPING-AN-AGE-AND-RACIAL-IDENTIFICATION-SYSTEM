
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import os
import csv
from classification.multi_head import MultiHeadClassifier
from classification.dataset import UTKFaceDataset
from classification.loss import focal_loss, cross_entropy_loss
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Dataset
dataset = UTKFaceDataset(img_dir='data/UTKFace')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Validation split (simple random split)
val_split = 0.1
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
model = MultiHeadClassifier(num_age_classes=7, num_race_classes=5, backbone_name="mobilenet_v2")
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Resume from last checkpoint if exists
start_epoch = 0
best_val_loss = float('inf')
ckpt_last = "model_last.pth"
ckpt_best = "model_best.pth"
log_csv = "classification/train_log.csv"
if os.path.exists(ckpt_last):
    checkpoint = torch.load(ckpt_last, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Resumed from {ckpt_last}, epoch {start_epoch}")

epochs = 20

for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0.0
    step = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, age_targets, race_targets in pbar:
        images = images.to(device)
        age_targets = age_targets.to(device)
        race_targets = race_targets.to(device)

        # Forward pass
        logits_age, logits_race = model(images)
        loss = cross_entropy_loss(logits_age, age_targets, logits_race, race_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1
        avg_loss = total_loss / step
        pbar.set_postfix({"loss_avg": f"{avg_loss:.4f}"})
    train_loss = total_loss / step
    # Validation
    model.eval()
    val_loss = 0.0
    val_steps = 0
    correct_age = 0
    correct_race = 0
    total = 0
    with torch.no_grad():
        for images, age_targets, race_targets in val_loader:
            images = images.to(device)
            age_targets = age_targets.to(device)
            race_targets = race_targets.to(device)
            logits_age, logits_race = model(images)
            loss = cross_entropy_loss(logits_age, age_targets, logits_race, race_targets)
            val_loss += loss.item()
            val_steps += 1
            pred_age = torch.argmax(logits_age, dim=1)
            pred_race = torch.argmax(logits_race, dim=1)
            correct_age += (pred_age == age_targets).sum().item()
            correct_race += (pred_race == race_targets).sum().item()
            total += images.size(0)
    val_loss = val_loss / max(1, val_steps)
    age_acc = correct_age / total if total > 0 else 0
    race_acc = correct_race / total if total > 0 else 0
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Age Acc: {age_acc:.4f}, Race Acc: {race_acc:.4f}")


    # Save last checkpoint (full)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, ckpt_last)

    # Save only model weights for inference
    torch.save(model.state_dict(), "model_last_only.pth")

    # Save best checkpoint (full)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, ckpt_best)
        # Save only model weights for best
        torch.save(model.state_dict(), "model_best_only.pth")
        print(f"Saved new best model at epoch {epoch+1}")

    # Append to CSV log
    file_exists = os.path.isfile(log_csv)
    # if file_exists is False: tạo file và ghi header 
    # if file_exists is True: chỉ ghi dữ liệu mới vào file đã có
    with open(log_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if (not file_exists) or os.stat(log_csv).st_size == 0:
            writer.writerow(["epoch", "train_loss", "val_loss", "age_acc", "race_acc"])
        writer.writerow([epoch+1, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{age_acc:.4f}", f"{race_acc:.4f}"])

def evaluate(model, dataloader, device):
    model.eval()
    correct_age = 0
    correct_race = 0
    total = 0
    with torch.no_grad():
        for images, age_targets, race_targets in dataloader:
            images = images.to(device)
            age_targets = age_targets.to(device)
            race_targets = race_targets.to(device)

            logits_age, logits_race = model(images)
            predicted_age = torch.argmax(logits_age, dim=1)
            predicted_race = torch.argmax(logits_race, dim=1)

            correct_age += (predicted_age == age_targets).sum().item()
            correct_race += (predicted_race == race_targets).sum().item()
            total += images.size(0)

        age_accuracy = correct_age / total if total > 0 else 0
        race_accuracy = correct_race / total if total > 0 else 0
    return age_accuracy, race_accuracy



        