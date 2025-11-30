"""
Transformer Model for Joint-Based Pointing Detection

Takes MediaPipe pose joints as input and predicts pointing direction.
Treats joints as a sequence and uses transformer encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return x


class JointTransformer(nn.Module):
    """
    Transformer model for pose joint-based pointing detection.
    
    Architecture:
    1. Project joint coordinates to embedding dimension
    2. Add positional encoding
    3. Transformer encoder layers
    4. Global pooling
    5. MLP heads for confidence and direction
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # x, y, z per joint
        num_joints: int = 33,  # MediaPipe pose has 33 landmarks
        d_model: int = 128,  # Embedding dimension
        nhead: int = 8,  # Number of attention heads
        num_layers: int = 4,  # Number of transformer layers
        dim_feedforward: int = 512,  # FFN dimension
        dropout: float = 0.1,
        pooling: str = 'cls'  # 'cls', 'mean', or 'max'
    ):
        """
        Args:
            input_dim: Dimension per joint (3 for x,y,z or 4 with visibility)
            num_joints: Number of joints (33 for MediaPipe)
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            pooling: Pooling strategy ('cls', 'mean', or 'max')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_joints = num_joints
        self.d_model = d_model
        self.pooling = pooling
        
        # Input projection: (batch, num_joints, input_dim) -> (batch, num_joints, d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # CLS token (if using cls pooling)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_joints + 1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # We'll use (seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output heads
        self.dropout = nn.Dropout(dropout)
        
        # Confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Direction head
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"JointTransformer initialized:")
        print(f"  - Input: {num_joints} joints × {input_dim} coords = {num_joints * input_dim} features")
        print(f"  - Embedding dim: {d_model}")
        print(f"  - Attention heads: {nhead}")
        print(f"  - Transformer layers: {num_layers}")
        print(f"  - Pooling: {pooling}")
        print(f"  - Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: Joint features of shape (batch, num_joints * input_dim)
               OR (batch, num_joints, input_dim)
        
        Returns:
            confidence: (batch, 1)
            direction: (batch, 3)
        """
        batch_size = x.size(0)
        
        # Reshape if needed: (batch, num_joints * input_dim) -> (batch, num_joints, input_dim)
        if x.dim() == 2:
            x = x.view(batch_size, self.num_joints, self.input_dim)
        
        # Project to embedding dimension: (batch, num_joints, input_dim) -> (batch, num_joints, d_model)
        x = self.input_projection(x)
        
        # Add CLS token if using cls pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_joints+1, d_model)
        
        # Change to (seq_len, batch, d_model) for transformer
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        
        # Pooling
        if self.pooling == 'cls':
            # Use CLS token
            pooled = x[0]  # (batch, d_model)
        elif self.pooling == 'mean':
            # Mean pooling
            pooled = x.mean(dim=0)  # (batch, d_model)
        elif self.pooling == 'max':
            # Max pooling
            pooled = x.max(dim=0)[0]  # (batch, d_model)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Output heads
        confidence = self.conf_head(pooled)  # (batch, 1)
        direction = self.dir_head(pooled)  # (batch, 3)
        
        return confidence, direction


class SimpleJointMLP(nn.Module):
    """
    Simple MLP baseline for comparison.
    Much faster than transformer but may be less accurate.
    """
    
    def __init__(
        self,
        input_dim: int = 99,  # 33 joints × 3 coords
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        self.conf_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.dir_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
        print(f"SimpleJointMLP initialized:")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Hidden dims: {hidden_dims}")
        print(f"  - Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Args:
            x: Joint features of shape (batch, input_dim)
        
        Returns:
            confidence: (batch, 1)
            direction: (batch, 3)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Output heads
        confidence = self.conf_head(features)
        direction = self.dir_head(features)
        
        return confidence, direction


# Factory functions
def create_joint_transformer(
    input_dim: int = 3,
    num_joints: int = 33,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1
):
    """Create joint transformer model"""
    return JointTransformer(
        input_dim=input_dim,
        num_joints=num_joints,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )


def create_simple_joint_mlp(
    input_dim: int = 99,
    hidden_dims: list = [512, 256, 128],
    dropout: float = 0.3
):
    """Create simple MLP baseline"""
    return SimpleJointMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )


if __name__ == "__main__":
    print("="*70)
    print("TESTING JOINT-BASED MODELS")
    print("="*70)
    
    batch_size = 4
    num_joints = 33
    input_dim = 3
    
    # Create dummy input
    x = torch.randn(batch_size, num_joints * input_dim)
    
    # Test Transformer
    print("\n1. Testing Joint Transformer:")
    print("-"*70)
    model_transformer = create_joint_transformer(
        input_dim=input_dim,
        num_joints=num_joints,
        d_model=128,
        nhead=8,
        num_layers=4
    )
    
    with torch.no_grad():
        conf, direction = model_transformer(x)
    
    print(f"\n✓ Output shapes:")
    print(f"  - Confidence: {conf.shape}")
    print(f"  - Direction: {direction.shape}")
    
    # Test MLP
    print("\n2. Testing Simple MLP:")
    print("-"*70)
    model_mlp = create_simple_joint_mlp(
        input_dim=num_joints * input_dim,
        hidden_dims=[512, 256, 128]
    )
    
    with torch.no_grad():
        conf, direction = model_mlp(x)
    
    print(f"\n✓ Output shapes:")
    print(f"  - Confidence: {conf.shape}")
    print(f"  - Direction: {direction.shape}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    
    # Compare model sizes
    print("\nModel Comparison:")
    print(f"  Transformer: {sum(p.numel() for p in model_transformer.parameters()):,} parameters")
    print(f"  Simple MLP:  {sum(p.numel() for p in model_mlp.parameters()):,} parameters")