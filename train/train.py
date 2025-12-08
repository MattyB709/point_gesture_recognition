import torch
from torchvision import models, transforms
from pointing_dataset import PointingDataset 
from vit_dataset import ViTDataset, ViTDatasetAggressive
import mediapipe_dataset
from joint_transformer import create_joint_transformer, create_simple_joint_mlp
from torch.utils.data import DataLoader
from torch import optim
from metrics import AngularLoss, angular_error
from tqdm import tqdm
import numpy as np
import wandb
from datetime import datetime
from torch.amp import GradScaler

# use for early stopping, if val loss doesn't decrease for PATIENCE epochs, kill the run
PATIENCE = 50

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp = False):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_conf_loss = 0.0
    total_vec_loss = 0.0
    angle_count = 0
    angular_error_deg = 0

    for batch_idx, (imgs, label_dict) in enumerate(dataloader):
        # Move to device
        imgs = imgs.to(device)

        # Prepare labels
        is_pointing = label_dict['is_pointing'].to(device).unsqueeze(1).float()
        vector = label_dict['pointing_vector'].to(device).float()

        angle_count += (is_pointing == 1.0).sum().item()
        # Forward pass
        with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = use_amp):
            outputs = model(imgs)
            
            # split outputs, first index is confidence, rest is vector
            pred_confidence = outputs[:, :1]
            # pred_confidence = torch.zeros((imgs.shape[0], 1)).to(device).requires_grad_(True)
            pred_vector = outputs[:, 1:]
            

            # Compute loss
            loss, conf_loss, vec_loss = criterion(pred_confidence, pred_vector, is_pointing, vector)
        mask = (is_pointing == 1.0).squeeze()
        if mask.sum() > 0:
            angular_error_deg += angular_error(pred_vector[mask], vector[mask]) * mask.sum().item()
            angle_count += mask.sum().item()
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        total_loss += loss.item()
        total_conf_loss += conf_loss.item()
        total_vec_loss += vec_loss.item() * (is_pointing == 1.0).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, Conf={conf_loss.item():.4f}, Vec={vec_loss.item():.4f}")

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_conf_loss = total_conf_loss / len(dataloader)
    avg_vec_loss = total_vec_loss / max(angle_count, 1)
    avg_angular_error = angular_error_deg / angle_count

    return avg_loss, avg_conf_loss, avg_vec_loss, avg_angular_error


def validate(model, dataloader, criterion, device, use_amp = False):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    total_conf_loss = 0.0
    total_vec_loss = 0.0
    angular_error_deg = 0.0
    angle_count = 0

    # Metrics
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, label_dict in dataloader:
            # Move to device
            imgs = imgs.to(device)

            # Prepare labels

            # confidence is shape (B, 1) while label_dict['is_pointing'] is (B,), so unsqueeze at 1
            # to match shapes
            is_pointing = label_dict['is_pointing'].to(device).unsqueeze(1).float()
            vector = label_dict['pointing_vector'].to(device).float()

            # Forward pass

            with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = use_amp):
                outputs = model(imgs)
                pred_confidence = outputs[:, :1]
                pred_vector = outputs[:, 1:]
                # pred_confidence = torch.zeros((imgs.shape[0], 1)).to(device).requires_grad_(True)
                # pred_vector= outputs
                # Compute loss
                loss, conf_loss, vec_loss = criterion(pred_confidence, pred_vector, is_pointing, vector)

            mask = (is_pointing == 1.0).squeeze()
            if mask.sum() > 0:
                angular_error_deg += angular_error(pred_vector[mask], vector[mask]) * mask.sum().item()
                angle_count += mask.sum().item()

            # Accumulate losses
            total_loss += loss.item()
            total_conf_loss += conf_loss.item()
            total_vec_loss += vec_loss.item() * mask.sum().item()

            # Classification accuracy
            pred_confidence = torch.sigmoid(pred_confidence)
            pred_class = (pred_confidence > 0.5).float()
            true_class = (is_pointing > 0.5).float()
            correct += (pred_class == true_class).sum().item()
            total += is_pointing.size(0)

    # Average losses and accuracy
    avg_loss = total_loss / len(dataloader)
    avg_conf_loss = total_conf_loss / len(dataloader)
    avg_vec_loss = total_vec_loss / max(angle_count, 1)
    accuracy = 100.0 * correct / total
    avg_angular_error = angular_error_deg / max(angle_count, 1)

    return avg_loss, avg_conf_loss, avg_vec_loss, accuracy, avg_angular_error

def train_model(model_name, train_loader, val_loader, num_epochs=50, lr=1e-4, device='cuda', use_wandb=False, use_amp=True, notes = "", aux_name = ""):
    """
    Complete training loop

    Args:
        model_name: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    """
    model = create_model(model_name)
    model = model.to(device)

    if use_wandb:
        formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        run_name = f"{model_name}_aug{train_loader.dataset.augment}_amp{use_amp}_{aux_name}_{formatted_date}"
        run = wandb.init(project="pointing_gesture_recognition", name = run_name, notes = notes,
                         config={"num_epochs": num_epochs, 
                                 "learning_rate": lr, 
                                 "batch_size": train_loader.batch_size, 
                                 "model": model_name,
                                 "train_samples": len(train_loader.dataset),
                                 "val_samples": len(val_loader.dataset),
                                 "augmentation": train_loader.dataset.augment,
                                 "use_amp": use_amp
                                 })

    # Loss and optimizer
    criterion = AngularLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scaler = GradScaler(enabled=use_amp)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=5)

    best_val_angular_error = float('inf')
    num_epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_conf_loss, train_vec_loss, train_angular_error = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler,use_amp = use_amp
        )

        print(f"Train Loss: {train_loss:.4f} (Conf: {train_conf_loss:.4f}, Vec: {train_vec_loss:.4f})")

        # Validate
        val_loss, val_conf_loss, val_vec_loss, val_acc, val_angular_error = validate(
            model, val_loader, criterion, device, use_amp = use_amp
        )

        if use_wandb:
            run.log({
                "Train Loss": train_loss,
                "Train Conf Loss": train_conf_loss,
                "Train Vec Loss": train_vec_loss,
                "Train Angular Error": train_angular_error,
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
        if val_angular_error < best_val_angular_error:
            best_val_angular_error = val_angular_error
            torch.save({
                'model_state_dict': model.state_dict(),
            }, f'trained_models/{run_name}.pth')
            print("✓ Saved best model")
            num_epochs_without_improvement = 0
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement >= PATIENCE:
                break
    
    if use_wandb:
        run.finish()

def create_resnet_frozen(
    model_name="ResNet50",
    freeze_until_layer=2,
    dropout=0.5
):
    """
    Create ResNet with frozen layers.
    
    Args:
        model_name: "ResNet18", "ResNet34", "ResNet50", "ResNet101"
        freeze_until_layer: 0-4 (how many layers to freeze)
        dropout: Dropout probability
    
    Returns:
        model: ResNet with frozen backbone and custom FC head
    """
    
    # Create base model
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "ResNet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == "ResNet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze layers
    if freeze_until_layer >= 1:
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
    
    if freeze_until_layer >= 2:
        for param in model.layer2.parameters():
            param.requires_grad = False
    
    if freeze_until_layer >= 3:
        for param in model.layer3.parameters():
            param.requires_grad = False
    
    if freeze_until_layer >= 4:
        for param in model.layer4.parameters():
            param.requires_grad = False
    
    # Replace FC head
    model.fc = model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 512),   # Compress and combine features
        torch.nn.ReLU(),              # Non-linearity
        torch.nn.Dropout(dropout),        # Regularization
        torch.nn.Linear(512, 4)       # Final output
    )
    
    return model

# function to create model from a set of prespecified names
def create_model(model_name: str):
    if model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 1 for confidence + 3 for vector
    elif model_name == "ViT_B_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)

        print("Freezing first 8 of 12 transformer blocks...")
        for i, block in enumerate(model.encoder.layers):
            if i < 8:
                for param in block.parameters():
                    param.requires_grad = False

        model.heads.head = torch.nn.Sequential(
            torch.nn.Dropout(0.5),  # ← Add this!
            torch.nn.Linear(model.heads.head.in_features, 4)
        )
    elif model_name == "ResNet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 1 for confidence + 3 for vector
    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # for param in model.parameters():
        #     param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 1 for confidence + 3 for vector
        # model.fc = torch.nn.Sequential(
        #     torch.nn.Dropout(0.5),  # ← Add this!
        #     torch.nn.Linear(model.fc.in_features, 256),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.Linear(256, 4),
        # )
        # model = create_resnet_frozen(model_name, 3, 0.5)
    elif model_name == "ResNet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 1 for confidence + 3 for vector
    elif model_name == "joint_transformer":
        model = create_joint_transformer()
    elif model_name == "mlp":
        model = create_simple_joint_mlp()
    else:
        raise Exception(f"Model name not found")
    return model


def create_pointing_transforms_v2(target_size=224):
    """
    Simple resize approach - works well for most cases.
    Your images are 1920x1080, so slight distortion is minimal.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size), 
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

class ResizeWithPad:
    """Custom transform that resizes maintaining aspect ratio and pads to square"""
    def __init__(self, target_size=224):
        self.target_size = target_size
    
    def __call__(self, img):
        # img is PIL Image
        w, h = img.size
        
        # Calculate new size maintaining aspect ratio
        if w > h:
            new_w = self.target_size
            new_h = int(h * self.target_size / w)
        else:
            new_h = self.target_size
            new_w = int(w * self.target_size / h)
        
        # Resize
        img = transforms.functional.resize(img, (new_h, new_w), 
                                          interpolation=transforms.InterpolationMode.BILINEAR)
        
        # Calculate padding
        pad_left = (self.target_size - new_w) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_top = (self.target_size - new_h) // 2
        pad_bottom = self.target_size - new_h - pad_top
        
        # Apply padding
        img = transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), 
                                       fill=0, padding_mode='constant')
        
        return im101g


def create_pointing_transforms_with_padding(target_size=224):
    """
    Better quality: Resize maintaining aspect ratio, then pad to square.
    Use this if you want to preserve aspect ratio exactly.
    """
    return transforms.Compose([
        ResizeWithPad(target_size=target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


if __name__ == "__main__":

    # Example usage 
    custom_transforms = create_pointing_transforms_with_padding(target_size=224)
    weights = models.ViT_B_16_Weights.transforms

    data_dir = "./split_data"
    # train_dataset = ViTDataset(
    #     "./split_data/train",
    #     transform=custom_transforms,
    #     augment=True  # Augments direction vectors
    # )
    
    # val_dataset = ViTDataset(
    #     "./split_data/val",
    #     transform=custom_transforms,
    #     augment=False
    # )
    # train_dataset = ViTDatasetAggressive(
    #     "./split_data/train",
    #     transform=custom_transforms,
    #     augment=True  # Augments direction vectors
    # )
    
    # val_dataset = ViTDatasetAggressive(
    #     "./split_data/val",
    #     transform=custom_transforms,
    #     augment=False
    # )
    # train_dataset = PointingDataset(data_dir + "/train", augment =False, normalize=False, transform=custom_transforms)
    # val_dataset = PointingDataset(data_dir + "/val", augment = False, normalize=False, transform=custom_transforms)

    # Pose Data Transformer
    # K = np.array([[919.76178, 0,     962.6875],
    #           [0,        919.8909, 550.9944],
    #           [0,        0,        1]], dtype=np.float64)
    
    # train_dataset = mediapipe_dataset.create_train_dataset(
    #     "./split_data/train",
    #     use_kinect_depth=True,
    #     K=K
    # )

    # val_dataset = mediapipe_dataset.create_val_dataset(
    #     "./split_data/val",
    #     use_kinect_depth=True,
    #     K=K
    # )

    # train_dataset.precompute_all_joints()
    # val_dataset.precompute_all_joints()

    # model = create_joint_transformer()

    train_dataset = PointingDataset(data_dir + "/train", augment = True, normalize=True)
    val_dataset = PointingDataset(data_dir + "/val", augment = False, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # model_name = "ResNet18"
    # train_model(model_name, train_loader, val_loader, num_epochs=200, lr=1e-4, device='cuda', use_wandb=True, use_amp=True, notes="Not pretrained")
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    model_name = "ResNet50"
    train_model(model_name, train_loader, val_loader, num_epochs=200, lr=1e-5, device='cuda', use_wandb=True, use_amp=True, 
                notes="training with cleaned data", aux_name="new_data")
