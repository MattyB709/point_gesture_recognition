import torch
from .pointing_dataset import PointingDataset 
from torch.utils.data import DataLoader
from torch import optim
from .angular_loss import AngularLoss

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_conf_loss = 0.0
    total_vec_loss = 0.0

    for batch_idx, (rgbd, label_dict) in enumerate(dataloader):
        # Move to device
        rgbd = rgbd.to(device)

        # Prepare labels
        batch_size = rgbd.size(0)
        confidence = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        vector = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)

        for i in range(batch_size):
            confidence[i, 0] = float(label_dict['is_pointing'][i])

            if label_dict['wrist_coords'][i] is not None:
                wrist = label_dict['wrist_coords'][i]
                pointing = label_dict['pointing_vector'][i]
                vector[i, :3] = wrist.to(device)
                vector[i, 3:] = pointing.to(device)

        # Forward pass
        pred_confidence, pred_vector = model(rgbd)

        # Compute loss
        loss, conf_loss, vec_loss = criterion(pred_confidence, pred_vector, confidence, vector)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_conf_loss += conf_loss.item()
        total_vec_loss += vec_loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, Conf={conf_loss.item():.4f}, Vec={vec_loss.item():.4f}")

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_conf_loss = total_conf_loss / len(dataloader)
    avg_vec_loss = total_vec_loss / len(dataloader)

    return avg_loss, avg_conf_loss, avg_vec_loss


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    total_conf_loss = 0.0
    total_vec_loss = 0.0

    # Metrics
    correct = 0
    total = 0

    with torch.no_grad():
        for rgbd, label_dict in dataloader:
            # Move to device
            rgbd = rgbd.to(device)

            # Prepare labels
            batch_size = rgbd.size(0)
            confidence = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
            vector = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)

            for i in range(batch_size):
                confidence[i, 0] = float(label_dict['is_pointing'][i])

                if label_dict['wrist_coords'][i] is not None:
                    wrist = label_dict['wrist_coords'][i]
                    pointing = label_dict['pointing_vector'][i]
                    vector[i, :3] = wrist.to(device)
                    vector[i, 3:] = pointing.to(device)

            # Forward pass
            pred_confidence, pred_vector = model(rgbd)

            # Compute loss
            loss, conf_loss, vec_loss = criterion(pred_confidence, pred_vector, confidence, vector)

            # Accumulate losses
            total_loss += loss.item()
            total_conf_loss += conf_loss.item()
            total_vec_loss += vec_loss.item()

            # Classification accuracy
            pred_class = (pred_confidence > 0.5).float()
            true_class = (confidence > 0.5).float()
            correct += (pred_class == true_class).sum().item()
            total += confidence.size(0)

    # Average losses and accuracy
    avg_loss = total_loss / len(dataloader)
    avg_conf_loss = total_conf_loss / len(dataloader)
    avg_vec_loss = total_vec_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, avg_conf_loss, avg_vec_loss, accuracy

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, device='cuda'):
    """
    Complete training loop

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    """
    model = model.to(device)

    # Loss and optimizer
    criterion = AngularLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=5)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_conf_loss, train_vec_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        print(f"Train Loss: {train_loss:.4f} (Conf: {train_conf_loss:.4f}, Vec: {train_vec_loss:.4f})")

        # Validate
        val_loss, val_conf_loss, val_vec_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(f"Val Loss: {val_loss:.4f} (Conf: {val_conf_loss:.4f}, Vec: {val_vec_loss:.4f})")
        print(f"Val Accuracy: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print("✓ Saved best model")



