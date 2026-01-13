import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from library import *
import torch.optim as optim
import os

# ---Setting---

BATCH_SIZE = 64
HEAD_ONLY_EPOCHS = 40
PARTIAL_FREEZE_EPOCHS = 60
FINE_TUNE_EPOCHS = 25

def train_model(model, train_loader, val_loader, device, epochs, stage):

    if stage == 0:
        model.unfreeze_backbone(last_n_blocks=1)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=1e-4
        )

    elif stage == 1:
        model.unfreeze_backbone(last_n_blocks=3)

        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = optim.Adam(
            [
                {'params': backbone_params, 'lr': 1e-5},
                {'params': head_params, 'lr': 5e-4}
            ],
            weight_decay=1e-4
        )

    else: # stage == 2
        model.unfreeze_backbone()

        optimizer = optim.Adam(
            model.parameters(),
            lr=5e-6,
            weight_decay=1e-4
        )

    LOSS_WEIGHTS = torch.tensor([1.0, 1.0], device=device)
    criterion = nn.SmoothL1Loss(beta=0.05, reduction="none")
    
    scaler = torch.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        train_loss_x = 0.0
        train_loss_y = 0.0

        for images, head_rotations, head_positions, head_depths, labels in train_loader:
            images = images.to(device) # (batch_size, 3, 96, 96)
            head_rotations = head_rotations.to(device) # (batch_size, 3)
            head_positions = head_positions.to(device) #(batch_size, 2)
            head_depths = head_depths.to(device) # (batch_size, 1)
            head_input = torch.cat([head_rotations, head_positions, head_depths], dim=1) # (batch_size, 6)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type):
                outputs = model(images, head_input)

                raw_loss = criterion(outputs, labels)
                weighted_loss = raw_loss * LOSS_WEIGHTS
                loss = weighted_loss.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_loss_x += raw_loss[:, 0].mean().item()
            train_loss_y += raw_loss[:, 1].mean().item()

        train_loss /= len(train_loader)
        train_loss_x /= len(train_loader)
        train_loss_y /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_loss_x = 0.0
        val_loss_y = 0.0

        with torch.no_grad():
            for images, head_rotations, head_positions, head_depths, labels in val_loader:
                images = images.to(device)
                head_rotations = head_rotations.to(device)
                head_positions = head_positions.to(device) 
                head_depths = head_depths.to(device)
                head_input = torch.cat([head_rotations, head_positions, head_depths], dim=1) # (batch_size, 6)
                labels = labels.to(device)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images, head_input)

                    raw_loss = criterion(outputs, labels)
                    weighted_loss = raw_loss * LOSS_WEIGHTS
                    loss = weighted_loss.mean()

                val_loss += loss.item()
                val_loss_x += raw_loss[:, 0].mean().item()
                val_loss_y += raw_loss[:, 1].mean().item()

        val_loss /= len(val_loader)
        val_loss_x /= len(val_loader)
        val_loss_y /= len(val_loader)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"  Train Loss: {train_loss:.6f} "
            f"(x: {train_loss_x:.6f}, y: {train_loss_y:.6f})  "
            f"  Val Loss: {val_loss:.6f} "
            f"(x: {val_loss_x:.6f}, y: {val_loss_y:.6f}) "
            f"  LR: {optimizer.param_groups[0]['lr']:.2e} "
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_gaze_model.pth")
            print("Saved best model.")
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    for param in model.parameters():
        param.grad = None

    return best_val_loss


def main():
    try:
        torch.backends.cudnn.benchmark = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = NeuralNetworkModel().to(device)

        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("\nTraining")
        train_dataset = GazeDataset(
            root="data/training", 
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        print("\nValidation")
        val_dataset = GazeDataset(
            root="data/validation",
            transform=transform
        )
        print("\n")
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        print("\nStarting training...\n")
        best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=HEAD_ONLY_EPOCHS,
            stage=0
        )

        print("\nStarting partial freeze...\n")
        best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=PARTIAL_FREEZE_EPOCHS,
            stage=1
        )

        print("\nStarting fine-tuning...\n")
        best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=FINE_TUNE_EPOCHS,
            stage=2
        )

    except KeyboardInterrupt:
        print("Stopping")
    
    finally:
        # save best model in model folder
        os.makedirs("saved models", exist_ok=True)
        model_idx = os.listdir("saved models")
        torch.save(torch.load("best_gaze_model.pth"), f"saved models/gaze_model{len(model_idx)}.pth")
        print(f"{best_val_loss:.6f}")

if __name__ == "__main__":
    main()
