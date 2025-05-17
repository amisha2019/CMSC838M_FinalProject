import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsEncoder(nn.Module):
    def __init__(self, out_dim: int = 4):
        super().__init__()
        # Backbone for encoding point clouds
        self.backbone = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Global pooling and final output head
        self.pooling = lambda x: torch.max(x, dim=2, keepdim=False)[0]
        self.head = nn.Linear(256, out_dim)
        
        # —– PHYSICS-SUPERVISION DECODER —–
        self.out_dim = out_dim  # Store for decoder
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the physics encoder
        
        Args:
            pc: Point cloud tensor of shape [B, N, 3]
                B = batch size, N = number of points
                
        Returns:
            phys: Physics encoding of shape [B, out_dim]
        """
        # Remove debug print statements in production code
        # print(f"PhysicsEncoder input shape: {pc.shape}")
        
        # Ensure point cloud is in the correct shape [B, 3, N]
        if pc.dim() == 3:
            pc = pc.transpose(1, 2)  # [B, N, 3] -> [B, 3, N]
            
        # print(f"Transposed shape: {pc.shape}")
        
        # Extract features
        x = self.backbone(pc)  # [B, 256, N]
        x = self.pooling(x)    # [B, 256]
        phys = self.head(x)    # [B, out_dim]
        
        return phys 
    
    def decode(self, phys_latent):
        """
        Decodes the physics latent vector to the physics vector space.
        This is a separate method to handle the case where we want to decode
        without running the full encoder.
        """
        return self.decoder(phys_latent) 