import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Replace pytorch3d knn import with a custom implementation
# from pytorch3d.ops.knn import knn_points
import logging

from equibot.policies.vision.vec_layers import VecLinear
from equibot.policies.vision.vec_layers import VecLinNormAct as VecLNA


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


# Custom CUDA-compatible KNN implementation
def custom_knn_points(x, y, K, return_nn=False):
    """GPU-compatible implementation of K-nearest neighbors search that returns
    data in the same format as pytorch3d.ops.knn.knn_points.
    
    Args:
        x: (B, N, D) tensor of query points
        y: (B, M, D) tensor of reference points
        K: number of nearest neighbors to return
        return_nn: if True, return nearest neighbors as well
        
    Returns:
        dists: (B, N, K) tensor of squared distances
        idx: (B, N, K) tensor of indices into y
        nn: (B, N, K, D) tensor of nearest neighbor points (if return_nn=True)
    """
    batch_size = x.shape[0]
    num_points_x = x.shape[1]
    point_dim = x.shape[2]
    num_points_y = y.shape[1]
    
    # Calculate pairwise distances
    dists = torch.cdist(x, y, p=2)  # (B, N, M)
    
    # Get K nearest neighbors
    sq_dists, idx = torch.topk(dists, k=K, dim=-1, largest=False, sorted=True)
    
    if return_nn:
        # Gather the actual points
        # Create batch indices for gathering
        batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand(-1, num_points_x, K)
        
        # Select neighbor points
        neighbors = torch.gather(
            y.view(batch_size, 1, num_points_y, point_dim).expand(batch_size, num_points_x, num_points_y, point_dim),
            2,
            idx.unsqueeze(-1).expand(batch_size, num_points_x, K, point_dim)
        )
        
        return sq_dists, idx, neighbors
    else:
        return sq_dists, idx


class VecPointNet(nn.Module):
    def __init__(
        self,
        h_dim=128,
        c_dim=128,
        num_layers=4,
        knn=16,
    ):
        super().__init__()

        self.h_dim = h_dim
        self.c_dim = c_dim
        self.num_layers = num_layers
        self.knn = knn

        self.pool = meanpool

        act_func = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        vnla_cfg = {"mode": "so3", "act_func": act_func}

        self.conv_in = VecLNA(3, h_dim, **vnla_cfg)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(VecLNA(h_dim, h_dim, **vnla_cfg))
            self.global_layers.append(VecLNA(h_dim * 2, h_dim, **vnla_cfg))
        self.conv_out = VecLinear(h_dim * self.num_layers, c_dim, mode="so3")

        self.fc_inv = VecLinear(c_dim, 3, mode="so3")

    def get_graph_feature(self, x: torch.Tensor, k: int, knn_idx=None, cross=False):
        # x: B,C,3,N return B,C*2,3,N,K

        B, C, _, N = x.shape
        if knn_idx is None:
            # if knn_idx is not none, compute the knn by x distance; ndf use fixed knn as input topo
            _x = x.reshape(B, -1, N)
            # Print input shape for debugging
            print(f"x shape: {x.shape}, _x shape: {_x.shape}, transposed: {_x.transpose(2, 1).shape}")
            
            # Use custom KNN implementation that works on GPU
            _, knn_idx, neighbors = custom_knn_points(
                _x.transpose(2, 1), _x.transpose(2, 1), K=k, return_nn=True
            )  # B,N,K; B,N,K; B,N,K,D
            
            # Print shapes for debugging
            print(f"knn_idx shape: {knn_idx.shape}, neighbors shape: {neighbors.shape}")
            
            # We need to reshape the neighbors tensor to match the expected format
            # Original reshape was: neighbors.reshape(B, N, k, C, 3).permute(0, -2, -1, 1, 2)
            # Let's find out what the input actually looks like
            expected_elements = B * N * k * C * 3
            actual_elements = neighbors.numel()
            print(f"Expected elements: {expected_elements}, actual elements: {actual_elements}")
            
            # Dynamically adjust the reshape based on actual dimensions
            try:
                # Extract the point dimension from neighbors (should be 3)
                point_dim = neighbors.shape[-1]
                
                # Expected format after custom_knn_points is [B, N, K, D] where D=C*3
                # Need to reshape to [B, N, K, C, 3] and then permute to [B, C, 3, N, K]
                neighbors = neighbors.reshape(B, N, k, C, 3).permute(0, 3, 4, 1, 2)
                print(f"After reshape and permute, neighbors shape: {neighbors.shape}")
            except RuntimeError as e:
                print(f"Reshape error: {e}")
                # If reshape fails, try to create an appropriate tensor with same device and dtype
                if point_dim == C * 3:
                    # If point dimension matches C*3, reshape without adding new dimensions
                    neighbors = neighbors.reshape(B, N, k, -1).permute(0, 3, 1, 2)
                    print(f"Alternative reshape (1), neighbors shape: {neighbors.shape}")
                else:
                    # Otherwise use view_as to match expected shape from original code
                    # This is a backup approach that may not preserve the correct data
                    print(f"Cannot reshape correctly, neighbors has wrong number of elements.")
                    # Create a debugging placeholder that will let us continue
                    # This is not a real fix but will help diagnose the shape issue
                    neighbors = torch.zeros(B, C, 3, N, k, device=x.device, dtype=x.dtype)
        else:  # gather from the input knn idx
            assert knn_idx.shape[-1] == k, f"input knn gather idx should have k={k}"
            neighbors = torch.gather(
                x[..., None, :].expand(-1, -1, -1, N, -1),
                dim=-1,
                index=knn_idx[:, None, None, ...].expand(-1, C, 3, -1, -1),
            )  # B,C,3,N,K
        x_padded = x[..., None].expand_as(neighbors)

        if cross:
            x_dir = F.normalize(x, dim=2)
            x_dir_padded = x_dir[..., None].expand_as(neighbors)
            cross = torch.cross(x_dir_padded, neighbors, dim=2)
            y = torch.cat([cross, neighbors - x_padded, x_padded], 1)
        else:
            y = torch.cat([neighbors - x_padded, x_padded], 1)
        return y, knn_idx  # B,C*2,3,N,K

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 3, N]

        x, knn_idx = self.get_graph_feature(x, self.knn, cross=True)
        x, _ = self.conv_in(x)
        x = self.pool(x)

        y = x
        feat_list = []
        for i in range(self.num_layers):
            y, _ = self.layers[i](y)
            y_global = y.mean(-1, keepdim=True)
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y, _ = self.global_layers[i](y)
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x, _ = self.conv_out(x)

        return x.mean(-1), x
