import torch
from torchvision import models
from pointing_dataset import PointingDataset 
from torch.utils.data import DataLoader
from torch import optim
from metrics import AngularLoss, angular_error
from tqdm import tqdm
import wandb

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_conf_loss = 0.0
    total_vec_loss = 0.0

    for batch_idx, (imgs, label_dict) in enumerate(dataloader):
        # Move to device
        imgs = imgs.to(device)

        # Prepare labels
        batch_size = imgs.size(0)
        confidence = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        vector = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)

        for i in range(batch_size):
            confidence[i, 0] = float(label_dict['is_pointing'][i])

            if confidence[i, 0] != 0.0:
                pointing = label_dict['pointing_vector'][i]
                vector[i, :] = pointing.to(device)

        # Forward pass
        outputs = model(imgs)
        
        # split outputs, first index is confidence, rest is vector
        pred_confidence = outputs[:, :1]
        pred_vector = outputs[:, 1:]

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
    angular_error_deg = 0.0

    # Metrics
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, label_dict in dataloader:
            # Move to device
            imgs = imgs.to(device)

            # Prepare labels
            batch_size = imgs.size(0)
            confidence = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
            vector = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)

            for i in range(batch_size):
                confidence[i, 0] = float(label_dict['is_pointing'][i])

                if confidence[i, 0] != 0.0:
                    pointing = label_dict['pointing_vector'][i]
                    vector[i, :] = pointing.to(device)

            # Forward pass
            outputs = model(imgs)
            pred_confidence = outputs[:, :1]
            pred_vector = outputs[:, 1:]

            # Compute loss
            loss, conf_loss, vec_loss = criterion(pred_confidence, pred_vector, confidence, vector)
            mask = (confidence == 1.0).squeeze()
            if mask.sum() > 0:
                angular_error_deg += angular_error(pred_vector[mask], vector[mask])

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
    avg_angular_error = angular_error_deg / len(dataloader)

    return avg_loss, avg_conf_loss, avg_vec_loss, accuracy, avg_angular_error

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, device='cuda', use_wandb=False):
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

    if use_wandb:
        run = wandb.init(project="pointing_gesture_recognition", 
                         config={"num_epochs": num_epochs, 
                                 "learning_rate": lr, 
                                 "batch_size": train_loader.batch_size, 
                                 "model": "ResNet18",
                                 "train_samples": len(train_loader.dataset),
                                 "val_samples": len(val_loader.dataset),
                                 "augmentation": True,
                                 })

    # Loss and optimizer
    criterion = AngularLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=5)

    best_val_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_conf_loss, train_vec_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        print(f"Train Loss: {train_loss:.4f} (Conf: {train_conf_loss:.4f}, Vec: {train_vec_loss:.4f})")

        # Validate
        val_loss, val_conf_loss, val_vec_loss, val_acc, val_angular_error = validate(
            model, val_loader, criterion, device
        )

        if use_wandb:
            run.log({
                "Train Loss": train_loss,
                "Train Conf Loss": train_conf_loss,
                "Train Vec Loss": train_vec_loss,
                "Val Loss": val_loss,
                "Val Conf Loss": val_conf_loss,
                "Val Vec Loss": val_vec_loss,
                "Val Accuracy": val_acc,
                "Val Angular Error": val_angular_error
            })

        print(f"Val Loss: {val_loss:.4f} (Conf: {val_conf_loss:.4f}, Vec: {val_vec_loss:.4f})")
        print(f"Val Accuracy: {val_acc:.2f}%")
        print(f"Val Angular Error: {val_angular_error:.2f} degrees")

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
    
    if use_wandb:
        run.finish()

def create_pointing_transforms_v2(target_size=224):
    """
    Better approach: Resize maintaining aspect ratio, then pad
    """
    return transforms.Compose([
        # Resize so that shortest side = target_size
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        
        # Pad to make square (preserves aspect ratio!)
        transforms.Pad(
            padding=lambda img: (
                (target_size - img.size[0]) // 2,  # left
                (target_size - img.size[1]) // 2,  # top
                (target_size - img.size[0] + 1) // 2,  # right
                (target_size - img.size[1] + 1) // 2,  # bottom
            ),
            fill=0,
            padding_mode='constant'
        ),
        
        # Ensure exactly target_size x target_size
        transforms.CenterCrop(target_size),
        
        transforms.ToTensor(),
        
        # ImageNet normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

if __name__ == "__main__":
    # Example usage
    # weights = models.ViT_B_16_Weights.DEFAULT
    # model = models.vit_b_16(weights=weights)
    # model.heads.head = torch.nn.Linear(model.heads.head.in_features, 4)  
    # transforms = weights.transforms()
    # custom_transforms = create_pointing_transforms_v2(target_size=224)
    data_dir = "./split_data"
    train_dataset = PointingDataset(data_dir + "/train")
    val_dataset = PointingDataset(data_dir + "/val")
    # train_dataset = PointingDataset(data_dir + "/train", transform=custom_transforms)
    # val_dataset = PointingDataset(data_dir + "/val", transform=custom_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Example model (replace with your actual model)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 1 for confidence + 3 for vector

    train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda', use_wandb=True)
    # train_model(
    #     model, 
    #     train_loader, 
    #     val_loader, 
    #     num_epochs=100, 
    #     lr=1e-5,  # Lower LR for ViT!
    #     device='cuda', 
    #     use_wandb=True
    # )
