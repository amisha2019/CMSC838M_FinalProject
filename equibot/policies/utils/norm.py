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
        nd = len(data.shape)
        target_shape = (1,) * (nd - 1) + (data.shape[-1],)
        try:
            dmin = self.stats["min"].reshape(target_shape)
            dmax = self.stats["max"].reshape(target_shape)
            return (data - dmin) / (dmax - dmin + 1e-12)
        except RuntimeError as e:
            # Handle shape mismatch by using a different approach
            print(f"Warning: Reshape failed in normalize. Applying adaptive normalization. Error: {e}")
            data_last_dim = data.shape[-1]

            dmin_for_op = self.stats["min"][:data_last_dim]
            dmax_for_op = self.stats["max"][:data_last_dim]

            op_dim = dmin_for_op.shape[-1]

            op_stats_reshaped_target = (1,) * (nd - 1) + (op_dim,)
            dmin_reshaped = dmin_for_op.reshape(op_stats_reshaped_target)
            dmax_reshaped = dmax_for_op.reshape(op_stats_reshaped_target)

            if op_dim < data_last_dim:
                # Stats are effectively smaller than data. Normalize only the corresponding slice of data.
                data_slice_to_norm = data[..., :op_dim]
                normalized_data_slice = (data_slice_to_norm - dmin_reshaped) / (dmax_reshaped - dmin_reshaped + 1e-12)
                
                result = data.clone() 
                result[..., :op_dim] = normalized_data_slice
                return result
            else:
                # op_dim == data_last_dim. Normalize the whole 'data' tensor.
                return (data - dmin_reshaped) / (dmax_reshaped - dmin_reshaped + 1e-12)

    def unnormalize(self, data):
        nd = len(data.shape)
        target_shape = (1,) * (nd - 1) + (data.shape[-1],)
        try:
            dmin = self.stats["min"].reshape(target_shape)
            dmax = self.stats["max"].reshape(target_shape)
            return data * (dmax - dmin) + dmin
        except RuntimeError as e:
            # Handle shape mismatch by using a different approach
            print(f"Warning: Reshape failed in unnormalize. Applying adaptive normalization. Error: {e}")
            data_last_dim = data.shape[-1]

            dmin_for_op = self.stats["min"][:data_last_dim]
            dmax_for_op = self.stats["max"][:data_last_dim]

            op_dim = dmin_for_op.shape[-1]

            op_stats_reshaped_target = (1,) * (nd - 1) + (op_dim,)
            dmin_reshaped = dmin_for_op.reshape(op_stats_reshaped_target)
            dmax_reshaped = dmax_for_op.reshape(op_stats_reshaped_target)

            if op_dim < data_last_dim:
                # Stats are effectively smaller than data. Unnormalize only the corresponding slice of data.
                data_slice_to_unnorm = data[..., :op_dim]
                unnormalized_data_slice = data_slice_to_unnorm * (dmax_reshaped - dmin_reshaped) + dmin_reshaped
                
                result = data.clone() 
                result[..., :op_dim] = unnormalized_data_slice
                return result
            else:
                # op_dim == data_last_dim. Unnormalize the whole 'data' tensor.
                return data * (dmax_reshaped - dmin_reshaped) + dmin_reshaped

    def state_dict(self):
        return self.stats

    def load_state_dict(self, state_dict):
        self.stats = state_dict
