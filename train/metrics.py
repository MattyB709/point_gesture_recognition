import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularLoss(nn.Module):
    """
    loss that combines binary cross-entropy for classification and
    a combination of MSE for position and cosine similarity for direction.
    This can be more appropriate for directional predictions
    """

    def __init__(self, alpha=1.0, beta=1.0, use_pos = False):
        super(AngularLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.use_pos = use_pos

    def forward(self, pred_confidence, pred_vector, true_confidence, true_vector):
        # Classification loss
        pred_confidence = torch.sigmoid(pred_confidence)
        conf_loss = self.bce_loss(pred_confidence, true_confidence)

        # Only compute vector loss for pointing samples
        pointing_mask = (true_confidence == 1.0).squeeze()

        if pointing_mask.sum() > 0:
            # Split into position and direction
            pred_dir = pred_vector[pointing_mask]
            true_dir = true_vector[pointing_mask]

            # Position loss (MSE)

            # Direction loss (cosine similarity)
            # Normalize vectors
            pred_dir_norm = F.normalize(pred_dir, p=2, dim=1)
            true_dir_norm = F.normalize(true_dir, p=2, dim=1)

            # Cosine similarity loss (1 - cos_sim)
            cos_sim = (pred_dir_norm * true_dir_norm).sum(dim=1).mean()
            dir_loss = 1 - cos_sim

        else:
            dir_loss = torch.tensor(0.0, device=pred_vector.device)

        total_loss = self.alpha * conf_loss + self.beta * dir_loss

        return total_loss, conf_loss, dir_loss

def angular_error(pred_vector, true_vector):
    """Compute angular error in degrees between predicted and true vectors"""
    # Normalize vectors
    pred_norm = F.normalize(pred_vector, p=2, dim=1)
    true_norm = F.normalize(true_vector, p=2, dim=1)

    # Compute cosine similarity
    # This dot product should give values between [-1, 1], but the clamp is to avoid numerical issues with arccos
    cos_sim = (pred_norm * true_norm).sum(dim=1).clamp(-1.0, 1.0)

    # Compute angle in radians and then convert to degrees
    angles_rad = torch.acos(cos_sim)
    angles_deg = angles_rad * (180.0 / torch.pi)

    return angles_deg.mean().item()