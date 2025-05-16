import torch, torch.nn as nn, torch.nn.functional as F

class PhysicsEncoder(nn.Module):
    """Predict a low‑dim vector of physical parameters from a point cloud."""
    def __init__(self, out_dim: int = 4):
        super().__init__()
        # tiny PointNet‑style encoder
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, 256)
        self.head = nn.Linear(256, out_dim)
        
        # Add decoder head for physics-supervision loss
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, out_dim)  # reconstruct same out_dim vector
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc shape: [B, N, 3]
        x = F.gelu(self.mlp1(pc))
        x = F.gelu(self.mlp2(x))
        x = F.gelu(self.mlp3(x))          # [B, N, 256]
        x = torch.max(x, dim=1).values    # global max‑pool
        return self.head(x)               # [B, out_dim] 