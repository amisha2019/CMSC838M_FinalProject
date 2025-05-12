import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.vision.physics_encoder import PhysicsEncoder
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D
from equibot.policies.utils.diffusion.diffusion_controller import DiffusionController


def vec2mat(vec):
    # converts non-orthogonal x, y vectors to a rotation matrix
    # input: (..., 2, 3)
    # output: (..., 3, 3)
    x, y = vec[..., [0], :], vec[..., [1], :]
    x_norm = F.normalize(x, dim=-1)
    y = y - x_norm * torch.sum(x_norm * y, dim=-1, keepdim=True)
    y_norm = F.normalize(y, dim=-1)
    z = torch.cross(x_norm, y_norm, dim=-1)
    mat = torch.cat([x_norm, y_norm, z], dim=-2).transpose(-2, -1)
    return mat


class EquiBotPolicy(nn.Module):
    def __init__(self, cfg, device="cpu"):
        nn.Module.__init__(self)
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.use_torch_compile = cfg.model.use_torch_compile
        self.device = device

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|       actions: 8
        # where |o| is the context observations and |a| are the predicted actions
        # The model will only look at the last |o| to predict the next |a|

        # Setup physics encoding if enabled
        if hasattr(cfg.model, 'use_physics_embed') and cfg.model.use_physics_embed:
            self.physics_enc = PhysicsEncoder(cfg.model.physics_embed_dim).to(device)
            self.physics_embed_dim = cfg.model.physics_embed_dim
        else:
            self.physics_enc = None
            self.physics_embed_dim = 0

        # Observation, action dimensions and types
        self.encoder = SIM3Vec4Latent(
            hidden_dim=cfg.model.hidden_dim, out_dim=cfg.model.hidden_dim * 3
        )
        
        # Add physics embedding dimension to the latent dimension
        latent_dim = cfg.model.hidden_dim * 3 + self.physics_embed_dim * 3
            
        if self.ac_mode == "diffusion":
            dim_in = 4 * 7  # 4 keypoints with 7 DOF each
            self.noise_pred_net = VecConditionalUnet1D(
                in_chans=dim_in,
                cond_chans=latent_dim,
                schedule_cfg=cfg.model.noise_scheduler,
            )
            self.noise_scheduler = DiffusionController(cfg.model.noise_scheduler, device)
        else:
            raise ValueError(f"Unknown ac_mode: {self.ac_mode}")

        self.nets = nn.ModuleDict(
            {"encoder": self.encoder, "noise_pred_net": self.noise_pred_net}
        )
        
        # Add physics encoder to ModuleDict if it exists
        if self.physics_enc is not None:
            self.nets["physics_enc"] = self.physics_enc

        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        if self.use_torch_compile:
            self.compiled = False

    def forward(self, obs, ema=True, physics_vec=None):
        """Forward pass through the policy.
        
        Args:
            obs: Dictionary of observations, which must include 'pc' and 'eef_pos'.
            ema: Whether to use the Exponential Moving Average model.
            physics_vec: Optional physics vector of shape [B, 4] containing:
                         [mass, friction, stiffness, damping] normalized to [0, 1].
                         If None but model.use_physics_embed=True, a default vector is used.
        
        Returns:
            q_k: Predicted joint positions for 4 keypoints, shape [B, 4, 7].
        """
        B = obs["pc"].shape[0]
        O = obs["pc"].shape[1]
        P = obs["pc"].shape[2]

        device = obs["pc"].device
        if O > 1:
            # multi-observation
            pc = obs["pc"].reshape(-1, P, 3)  # [B*O, P, 3]
            z_local = self.nets["encoder"](pc)  # [B*O, 3*hidden_dim]
            z_local = z_local.reshape(B, O, -1)  # [B, O, 3*hidden_dim]
            z = z_local[:, -1].reshape(B, 1, -1, 3)  # [B, 1, hidden_dim, 3]
        else:
            pc = obs["pc"].reshape(-1, P, 3)  # [B, P, 3]
            z = self.nets["encoder"](pc)  # [B, 3*hidden_dim]
            z = z.reshape(B, 1, -1, 3)  # [B, 1, hidden_dim, 3]

        if ema:
            ema_nets = self.ema.averaged_model
            z_local = ema_nets["encoder"](pc)  # [B, 3*hidden_dim]
            z = z_local.reshape(B, 1, -1, 3)  # [B, 1, hidden_dim, 3]
            
            # Add physics embedding if configured
            if self.physics_enc is not None:
                # Use provided physics vector or get default
                if physics_vec is None:
                    if "physics_vec" in obs:
                        physics_vec = obs["physics_vec"]
                        if isinstance(physics_vec, np.ndarray):
                            physics_vec = torch.from_numpy(physics_vec).to(device)
                    else:
                        physics_vec = torch.ones(B, 4, device=device) * 0.5  # default
                
                # Process PC to get physics embedding
                phys_latent = ema_nets["physics_enc"](pc[:B])  # [B, physics_embed_dim]
                
                # Reshape to match the z tensor dimensions
                phys_latent = phys_latent.unsqueeze(1).reshape(B, 1, -1, 1)
                # Expand to match the 3D dimension of other tensors
                phys_latent = phys_latent.expand(-1, -1, -1, 3)
                # Concatenate with observation embedding
                z = torch.cat([z, phys_latent], dim=2)  # [B, 1, hidden_dim+physics_embed_dim, 3]

        if self.ac_mode == "diffusion":
            Ho = 1
            
            # Prepare eef position for noise prediction
            q_k = torch.zeros(B, Ho, 4, 7, device=device)
            # Set the first step of the keypoints to the current eef position
            q_k[:, 0, :, :3] = obs["eef_pos"].reshape(B, 1, -1, 3)
            q_k[:, 0, :, 3:6] = torch.zeros([B, 1, 4, 3], device=device)
            q_k[:, 0, :, 6:] = torch.zeros([B, 1, 4, 1], device=device)

            # Shape before noise prediction: [B, Ho, 4, 7]
            # Reshape to [B, Ho, 28] for noise_pred_net
            q_k_flat = q_k.reshape(B, Ho, -1)
            z_flat = z.reshape(B, Ho, -1)
            
            if ema:
                noise_hat = ema_nets["noise_pred_net"](
                    torch.cat([q_k_flat, z_flat], dim=-1), t=torch.zeros(B, device=device)
                )
            else:
                noise_hat = self.nets["noise_pred_net"](
                    torch.cat([q_k_flat, z_flat], dim=-1), t=torch.zeros(B, device=device)
                )
                
            return q_k

    def step_ema(self):
        self.ema.step(self.nets)

    def sample(self, obs, ema=True, physics_vec=None):
        B = obs["pc"].shape[0]
        O = obs["pc"].shape[1]
        P = obs["pc"].shape[2]

        device = obs["pc"].device
        if O > 1:
            # multi-observation
            pc = obs["pc"].reshape(-1, P, 3)  # [B*O, P, 3]
            z_local = self.nets["encoder"](pc)  # [B*O, 3*hidden_dim]
            z_local = z_local.reshape(B, O, -1)  # [B, O, 3*hidden_dim]
            z = z_local[:, -1].reshape(B, 1, -1, 3)  # [B, 1, hidden_dim, 3]
        else:
            pc = obs["pc"].reshape(-1, P, 3)  # [B, P, 3]
            z = self.nets["encoder"](pc)  # [B, 3*hidden_dim]
            z = z.reshape(B, 1, -1, 3)  # [B, 1, hidden_dim, 3]

        if ema:
            ema_nets = self.ema.averaged_model
            z_local = ema_nets["encoder"](pc)  # [B, 3*hidden_dim]
            z = z_local.reshape(B, 1, -1, 3)  # [B, 1, hidden_dim, 3]
            
            # Add physics embedding if configured
            if self.physics_enc is not None:
                # Use provided physics vector or get default
                if physics_vec is None:
                    physics_vec = torch.ones(B, 4, device=device) * 0.5  # default
                elif isinstance(physics_vec, np.ndarray):
                    physics_vec = torch.from_numpy(physics_vec).to(device)
                
                # Get physics embedding
                phys_latent = ema_nets["physics_enc"](pc)  # [B, physics_embed_dim]
                # Reshape to match the z tensor dimensions
                phys_latent = phys_latent.unsqueeze(1).repeat(1, 1, 1).reshape(B, 1, -1, 1)
                # Expand to match the 3D dimension of other tensors
                phys_latent = phys_latent.expand(-1, -1, -1, 3)
                # Concatenate with observation embedding
                z = torch.cat([z, phys_latent], dim=2)  # [B, 1, hidden_dim+physics_embed_dim, 3]

        if self.ac_mode == "diffusion":
            Ho = 1
            x = torch.randn(
                [B, Ho, 4, 7], device=device
            )  # start from pure noise N(0, 1)

            # Set the first step of the keypoints to the current eef position
            x[:, 0, :, :3] = obs["eef_pos"].reshape(B, 1, -1, 3)

            # Shape before sampling: [B, Ho, 4, 7]
            # Reshape to [B, Ho, 28] for noise_pred_net
            x_flat = x.reshape(B, Ho, -1)
            z_flat = z.reshape(B, Ho, -1)
            ema_nets = self.ema.averaged_model
            
            # Generate samples from noise using DDPM
            if ema:
                def model_fn(x, t, cond):
                    if cond is not None:
                        # [B, Ho, C] + [B, Ho, cond_C] -> [B, Ho, C + cond_C]
                        model_input = torch.cat([x, cond], dim=-1)
                    else:
                        model_input = x
                    return ema_nets["noise_pred_net"](model_input, t=t)
            else:
                def model_fn(x, t, cond):
                    if cond is not None:
                        # [B, Ho, C] + [B, Ho, cond_C] -> [B, Ho, C + cond_C]
                        model_input = torch.cat([x, cond], dim=-1)
                    else:
                        model_input = x
                    return self.nets["noise_pred_net"](model_input, t=t)

            # Sample using diffusion model
            samples = self.noise_scheduler.p_sample_loop(
                model_fn, x_flat.shape, x_flat, z_flat
            )

            # Reshape to keypoint format: [B, Ho, 4, 7]
            q_k = samples.reshape(B, Ho, 4, 7)
            return q_k
