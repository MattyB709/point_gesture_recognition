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
        conf_loss = self.bce_loss(pred_confidence, true_confidence)

        # Only compute vector loss for pointing samples
        pointing_mask = (true_confidence > 0.5).squeeze()

        if pointing_mask.sum() > 0:
            # Split into position and direction
            pred_pos = pred_vector[pointing_mask, :3]
            pred_dir = pred_vector[pointing_mask, 3:]
            true_pos = true_vector[pointing_mask, :3]
            true_dir = true_vector[pointing_mask, 3:]

            # Position loss (MSE)
            if self.use_pos:
                pos_loss = self.mse_loss(pred_pos, true_pos)
            else:
                pos_loss = torch.tensor(0.0, device=pred_vector.device)

            # Direction loss (cosine similarity)
            # Normalize vectors
            pred_dir_norm = F.normalize(pred_dir, p=2, dim=1)
            true_dir_norm = F.normalize(true_dir, p=2, dim=1)

            # Cosine similarity loss (1 - cos_sim)
            cos_sim = (pred_dir_norm * true_dir_norm).sum(dim=1).mean()
            dir_loss = 1 - cos_sim

            vec_loss = pos_loss + dir_loss
        else:
            vec_loss = torch.tensor(0.0, device=pred_vector.device)

        total_loss = self.alpha * conf_loss + self.beta * vec_loss

        return total_loss, conf_loss, vec_loss