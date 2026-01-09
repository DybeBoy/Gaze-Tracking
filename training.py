import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from library import *
import torch.optim as optim
import os

# ---Setting---

# Main
BATCH_SIZE = 64
EPOCHS = 200
FREEZE_EPICHS = 20
LR = 1e-3

# Scheduler
SC_FACTOR = 0.5
SC_PATIENCE = 8
MIN_LR = 1e-6

# Early stop
ES_PATIENCE = 15


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

        criterion = nn.SmoothL1Loss(beta=0.05)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=LR,
            weight_decay=1e-4
        )
        scaler = torch.amp.GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=SC_FACTOR,
            patience=SC_PATIENCE,
            min_lr=MIN_LR,
        )

        best_val_loss = float("inf")
        early_stop_counter = 0
        early_stop_patience = ES_PATIENCE

        model.freeze_backbone(freeze=True)

        for epoch in range(EPOCHS):
            if epoch == FREEZE_EPICHS:
                model.freeze_backbone(freeze=False)

                print("Unfroze backbone for fine-tuning.")

            model.train()
            train_loss = 0.0

            for images, head_rotations, head_positions, labels in train_loader:
                images = images.to(device) # (batch_size, 3, 96, 96)
                head_rotations = head_rotations.to(device) # (batch_size, 3)
                head_positions = head_positions.to(device) #(batch_size, 2)
                #head_depths = head_depths.to(device) # (batch_size, 1)
                head_input = torch.cat([head_rotations, head_positions], dim=1) # (batch_size, 6)
                labels = labels.to(device)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images, head_input)
                    loss = criterion(outputs, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, head_rotations, head_positions, labels in val_loader:
                    images = images.to(device)
                    head_rotations = head_rotations.to(device)
                    head_positions = head_positions.to(device) 
                    #head_depths = head_depths.to(device)
                    head_input = torch.cat([head_rotations, head_positions], dim=1) # (batch_size, 6)
                    labels = labels.to(device)

                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(images, head_input)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{EPOCHS}: "
                f"  Train Loss: {train_loss:.6f} "
                f"  Val Loss: {val_loss:.6f} "
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
