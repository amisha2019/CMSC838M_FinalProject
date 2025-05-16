import numpy as np
import torch


class Normalizer(object):
    def __init__(self, data, symmetric=False, indices=None):
        if isinstance(data, dict):
            # load from existing data statistics
            self.stats = data
        elif symmetric:
            # just scaling applied in normalization, no bias
            # perform the same normalization in groups
            if indices is None:
                indices = np.arange(data.shape[-1])[None]

            self.stats = {
                "min": torch.zeros([data.shape[-1]]).to(data.device),
                "max": torch.ones([data.shape[-1]]).to(data.device),
            }
            for group in indices:
                max_abs = torch.abs(data[:, group]).max(0)[0].detach()
                limits = torch.ones_like(max_abs) * torch.max(max_abs)
                self.stats["max"][group] = limits
        else:
            mask = torch.zeros([data.shape[-1]]).to(data.device)
            if indices is not None:
                mask[indices.flatten()] += 1
            else:
                mask += 1
            self.stats = {
                "min": data.min(0)[0].detach() * mask,
                "max": data.max(0)[0].detach() * mask + 1.0 * (1 - mask),
            }

    def normalize(self, data):
        """Normalize data using stored statistics.
        
        Args:
            data: Input tensor to normalize
            
        Returns:
            Normalized tensor with same shape as input
        """
        # Get the shape of the input data
        nd = len(data.shape)
        data_size = data.shape[-1]
        
        # Make sure we're only using relevant stats
        min_vals = self.stats["min"]
        max_vals = self.stats["max"]
        
        # Handle dimension mismatch by truncating or padding if needed
        if data_size < min_vals.shape[0]:
            min_vals = min_vals[:data_size]
            max_vals = max_vals[:data_size]
        
        # Create target shape for broadcasting
        target_shape = tuple([1] * (nd - 1) + [data_size])
        
        # Safe reshape
        min_tensor = min_vals.clone().detach()
        max_tensor = max_vals.clone().detach()
        
        # Ensure min_tensor and max_tensor have the right size before reshaping
        if min_tensor.numel() != data_size:
            print(f"Warning: Stats size {min_tensor.numel()} != data size {data_size}. Adjusting...")
            if min_tensor.numel() > data_size:
                min_tensor = min_tensor[:data_size]
                max_tensor = max_tensor[:data_size]
            else:
                # Padding case - less likely but handle it
                padded_min = torch.zeros(data_size, device=min_tensor.device)
                padded_max = torch.ones(data_size, device=max_tensor.device)
                padded_min[:min_tensor.numel()] = min_tensor
                padded_max[:max_tensor.numel()] = max_tensor
                min_tensor = padded_min
                max_tensor = padded_max
        
        # Now reshape for broadcasting
        min_tensor = min_tensor.reshape(target_shape)
        max_tensor = max_tensor.reshape(target_shape)
        
        # Normalize using broadcasting
        return (data - min_tensor) / (max_tensor - min_tensor + 1e-12)

    def unnormalize(self, data):
        """Unnormalize data using stored statistics.
        
        Args:
            data: Input tensor to unnormalize
            
        Returns:
            Unnormalized tensor with same shape as input
        """
        # Get the shape of the input data
        nd = len(data.shape)
        data_size = data.shape[-1]
        
        # Make sure we're only using relevant stats
        min_vals = self.stats["min"]
        max_vals = self.stats["max"]
        
        # Handle dimension mismatch by truncating or padding if needed
        if data_size < min_vals.shape[0]:
            min_vals = min_vals[:data_size]
            max_vals = max_vals[:data_size]
        
        # Create target shape for broadcasting
        target_shape = tuple([1] * (nd - 1) + [data_size])
        
        # Safe reshape
        min_tensor = min_vals.clone().detach()
        max_tensor = max_vals.clone().detach()
        
        # Ensure min_tensor and max_tensor have the right size before reshaping
        if min_tensor.numel() != data_size:
            print(f"Warning: Stats size {min_tensor.numel()} != data size {data_size}. Adjusting...")
            if min_tensor.numel() > data_size:
                min_tensor = min_tensor[:data_size]
                max_tensor = max_tensor[:data_size]
            else:
                # Padding case - less likely but handle it
                padded_min = torch.zeros(data_size, device=min_tensor.device)
                padded_max = torch.ones(data_size, device=max_tensor.device)
                padded_min[:min_tensor.numel()] = min_tensor
                padded_max[:max_tensor.numel()] = max_tensor
                min_tensor = padded_min
                max_tensor = padded_max
        
        # Now reshape for broadcasting
        min_tensor = min_tensor.reshape(target_shape)
        max_tensor = max_tensor.reshape(target_shape)
        
        # Unnormalize using broadcasting
        return data * (max_tensor - min_tensor) + min_tensor

    def state_dict(self):
        return self.stats

    def load_state_dict(self, state_dict):
        self.stats = state_dict
