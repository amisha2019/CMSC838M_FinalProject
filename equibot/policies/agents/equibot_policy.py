import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.vision.physics_encoder import PhysicsEncoder
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D


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
        self.cfg = cfg  # Add cfg as instance variable for access
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
            c_dim=cfg.model.encoder.c_dim,
            backbone_type=cfg.model.encoder.backbone_type,
            backbone_args={
                "h_dim": cfg.model.hidden_dim,
                "c_dim": cfg.model.encoder.c_dim,
                "num_layers": cfg.model.encoder.backbone_args.num_layers,
                "knn": cfg.model.encoder.backbone_args.knn
            }
        )
        
        # Add physics embedding dimension to the latent dimension
        latent_dim = cfg.model.hidden_dim * 3 + self.physics_embed_dim * 3
            
        if self.ac_mode == "diffusion":
            dim_in = 4 * 7  # 4 keypoints with 7 DOF each
            self.noise_pred_net = VecConditionalUnet1D(
                input_dim=dim_in,
                cond_dim=latent_dim,
            )
            # Direct instantiation instead of using hydra
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.model.noise_scheduler.num_train_timesteps,
                beta_schedule=cfg.model.noise_scheduler.beta_schedule,
                clip_sample=cfg.model.noise_scheduler.clip_sample,
                prediction_type=cfg.model.noise_scheduler.prediction_type
            )
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
            
        # Add handle attributes for compatibility with agent code
        self.noise_pred_net_handle = self.noise_pred_net
        
    def _init_torch_compile(self):
        """Initialize torch.compile if enabled."""
        if self.use_torch_compile:
            try:
                import torch._dynamo as dynamo
                self.noise_pred_net_handle = torch.compile(self.noise_pred_net)
                self.compiled = True
            except Exception as e:
                print(f"Failed to compile model: {e}")
                self.compiled = False

    def _convert_state_to_vec(self, state):
        """Convert state tensor to vector representation for conditioning.
        
        Args:
            state: State tensor of shape [B, T, num_eef, eef_dim]
                where B is batch size, T is time steps, num_eef is number of end effectors,
                and eef_dim is the end effector dimension (typically 3 for xyz or 7 for pose)
                
        Returns:
            z_pos: Position vector of shape [B, T, num_eef, 3]
            z_dir: Direction vector of shape [B, T, num_eef, 3] or None if not using rotation
            z_scalar: Scalar values (e.g., gripper) of shape [B, T, num_eef] or None if not using gripper
        """
        B, T, num_eef, eef_dim = state.shape
        
        if eef_dim == 3:
            # Just xyz position
            z_pos = state
            z_dir = None
            z_scalar = None
        elif eef_dim == 4:
            # xyz position + gripper
            z_pos = state[..., 1:4]  # xyz is typically at indices 1,2,3
            z_dir = None
            z_scalar = state[..., 0:1]  # gripper is typically at index 0
        elif eef_dim == 7:
            # xyz position + rotation (quaternion) + gripper
            z_pos = state[..., 1:4]  # xyz is typically at indices 1,2,3
            z_dir = state[..., 4:7]  # rotation is typically at indices 4,5,6
            z_scalar = state[..., 0:1]  # gripper is typically at index 0
        else:
            raise ValueError(f"Unsupported eef_dim: {eef_dim}")
        
        return z_pos, z_dir, z_scalar
            
    def _convert_action_to_vec(self, action, batch=None):
        """Convert action tensor to vector representation for diffusion model.
        
        Args:
            action: Action tensor of shape [B, T, D] where D is the total action dimension
            batch: Optional batch dictionary with additional information
            
        Returns:
            vec_eef_action: End effector action vector of shape [B, T, num_eef, 3]
            vec_gripper_action: Gripper action vector of shape [B, T, num_eef] or None if not using gripper
        """
        B, T, D = action.shape
        
        # Calculate number of end effectors and DOF per end effector
        if hasattr(self.cfg.env, 'num_eef') and hasattr(self.cfg.env, 'dof'):
            num_eef = self.cfg.env.num_eef
            dof = self.cfg.env.dof
        else:
            # Default to 1 end effector and try to infer DOF from action dimension
            num_eef = 1
            dof = D // num_eef
            
        # Check if action size matches expected shape
        expected_size = B * T * num_eef * dof
        actual_size = action.numel()
        
        if expected_size != actual_size:
            print(f"Warning: Unexpected action size in _convert_action_to_vec. Expected {expected_size}, got {actual_size}")
            # Try to adapt to the actual data
            if actual_size % (B * T * num_eef) == 0:
                # Calculate the actual dof based on data
                actual_dof = actual_size // (B * T * num_eef)
                print(f"Adapting to action dof={actual_dof} instead of configured dof={dof}")
                dof = actual_dof
                
        try:
            action = action.reshape(B, T, num_eef, dof)
        except RuntimeError as e:
            print(f"Error reshaping action: {e}")
            print(f"Action shape: {action.shape}, trying alternative reshape")
            # If we can't reshape as expected, handle the special case
            # For this adaptation, we'll assume a 1D action vector and extract what we need
            if dof == 3:
                # Just xyz position
                # Extract first 3 values per step as position
                vec_eef_action = action.reshape(B, T, -1)[..., :3].unsqueeze(2)
                vec_gripper_action = None
                return vec_eef_action, vec_gripper_action
            elif dof == 4:
                # Gripper + xyz
                # Extract first value as gripper, next 3 as position
                reshaped = action.reshape(B, T, -1)
                vec_eef_action = reshaped[..., 1:4].unsqueeze(2)
                vec_gripper_action = reshaped[..., 0:1].unsqueeze(2)
                return vec_eef_action, vec_gripper_action
            else:
                # Just reshape to expected output shapes
                vec_eef_action = torch.zeros(B, T, num_eef, 3, device=action.device)
                vec_gripper_action = torch.zeros(B, T, num_eef, 1, device=action.device)
                return vec_eef_action, vec_gripper_action
        
        if dof == 3:
            # Just xyz position
            vec_eef_action = action
            vec_gripper_action = None
        elif dof == 4:
            # xyz position + gripper
            vec_eef_action = action[..., 1:4]  # xyz is typically at indices 1,2,3
            vec_gripper_action = action[..., 0:1]  # gripper is typically at index 0
        elif dof == 7:
            # xyz position + rotation (quaternion) + gripper
            vec_eef_action = action[..., 1:4]  # xyz is typically at indices 1,2,3
            vec_gripper_action = action[..., 0:1]  # gripper is typically at index 0
        else:
            # Handle arbitrary dof by assuming first value is gripper and next 3 are xyz
            # This is a fallback for unexpected dof values
            print(f"Using fallback handling for unusual dof value: {dof}")
            if dof > 3:
                vec_eef_action = action[..., 1:4]  # Assume indices 1,2,3 are xyz
                vec_gripper_action = action[..., 0:1]  # Assume index 0 is gripper
            else:
                vec_eef_action = action  # Use all values as position
                vec_gripper_action = None
            
        return vec_eef_action, vec_gripper_action
            
    def _convert_action_to_scalar(self, vec_eef_action, vec_gripper_action=None, batch=None):
        """Convert vector action representation back to scalar action.
        
        Args:
            vec_eef_action: End effector action vector of shape [B, T, num_eef, 3]
            vec_gripper_action: Optional gripper action vector of shape [B, T, num_eef, 1]
            batch: Optional batch dictionary with additional information
            
        Returns:
            action: Action tensor in scalar format
        """
        try:
            if vec_gripper_action is None:
                # Just position, no gripper
                return vec_eef_action
            else:
                # Position + gripper
                return torch.cat([vec_gripper_action, vec_eef_action], dim=-1)
        except RuntimeError as e:
            print(f"Error in _convert_action_to_scalar: {e}")
            print(f"vec_eef_action shape: {vec_eef_action.shape}")
            if vec_gripper_action is not None:
                print(f"vec_gripper_action shape: {vec_gripper_action.shape}")
            
            # Try to reshape tensors to be compatible
            B = vec_eef_action.shape[0]
            T = vec_eef_action.shape[1]
            
            if len(vec_eef_action.shape) == 4:  # [B, T, num_eef, 3]
                flattened_eef = vec_eef_action.reshape(B, T, -1)
                if vec_gripper_action is not None:
                    flattened_gripper = vec_gripper_action.reshape(B, T, -1)
                    return torch.cat([flattened_gripper, flattened_eef], dim=-1)
                else:
                    return flattened_eef
            else:
                # In case shape is unexpected, just return vec_eef_action
                return vec_eef_action

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
                
                # Add-on #3: Latent-dropout (stochastic FiLM)
                if self.training and hasattr(self.cfg.model, 'latent_dropout') and self.cfg.model.latent_dropout > 0.0:
                    mask = torch.rand_like(phys_latent[:, :1]) > self.cfg.model.latent_dropout
                    phys_latent = phys_latent * mask
                
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
                noise_hat = self.noise_pred_net_handle(
                    torch.cat([q_k_flat, z_flat], dim=-1), t=torch.zeros(B, device=device)
                )
            else:
                noise_hat = self.noise_pred_net(
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
            
            # Generate samples from noise using the scheduler
            if ema:
                def model_fn(x_t, t, context=None):
                    # Prepare input for noise_pred_net
                    if context is not None:
                        # [B, Ho, C] + [B, Ho, cond_C] -> [B, Ho, C + cond_C]
                        model_input = torch.cat([x_t, context], dim=-1)
                    else:
                        model_input = x_t
                    return self.noise_pred_net_handle(model_input, t=t)
            else:
                def model_fn(x_t, t, context=None):
                    if context is not None:
                        # [B, Ho, C] + [B, Ho, cond_C] -> [B, Ho, C + cond_C]
                        model_input = torch.cat([x_t, context], dim=-1)
                    else:
                        model_input = x_t
                    return self.noise_pred_net(model_input, t=t)

            # Use the scheduler's native sampling
            self.noise_scheduler.set_timesteps(1000, device=device)
            
            # Start with noise and gradually denoise
            curr_sample = x_flat
            for t in self.noise_scheduler.timesteps:
                # Get model prediction
                with torch.no_grad():
                    noise_pred = model_fn(curr_sample, t, z_flat)
                # Denoise one step
                curr_sample = self.noise_scheduler.step(noise_pred, t, curr_sample).prev_sample
            
            # Reshape back to keypoint format: [B, Ho, 4, 7]
            q_k = curr_sample.reshape(B, Ho, 4, 7)
            return q_k
