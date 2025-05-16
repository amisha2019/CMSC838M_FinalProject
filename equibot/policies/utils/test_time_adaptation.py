import torch
import torch.nn.functional as F

def apply_test_time_adaptation(model, obs, num_steps=3, lr=0.001):
    """
    Apply test-time adaptation to the physics encoder for 1-3 gradient steps.
    
    Args:
        model: The EquiBotPolicy model with physics_enc
        obs: Observation dictionary containing 'pc'
        num_steps: Number of gradient steps to take (1-3)
        lr: Learning rate for adaptation
    
    Returns:
        model: The adapted model with updated physics encoder
    """
    # Check if model has physics encoder
    if not hasattr(model, 'physics_enc') or model.physics_enc is None:
        return model
        
    # Prepare PC data
    if isinstance(obs["pc"], list):
        pc = torch.from_numpy(obs["pc"][-1]).to(next(model.parameters()).device).float()
    else:
        pc = torch.from_numpy(obs["pc"]).to(next(model.parameters()).device).float()
    
    # Add batch dimension if needed
    if len(pc.shape) == 2:
        pc = pc.unsqueeze(0)  # [1, N, 3]
    
    # Create optimizer for only the physics encoder
    optimizer = torch.optim.Adam(model.physics_enc.parameters(), lr=lr)
    
    # Enable gradient tracking for the physics encoder
    for param in model.physics_enc.parameters():
        param.requires_grad = True
    
    # Keep the rest of the model frozen
    for name, param in model.named_parameters():
        if 'physics_enc' not in name:
            param.requires_grad = False
    
    # Perform test-time adaptation
    model.physics_enc.train()  # Set physics encoder to training mode
    
    with torch.enable_grad():
        for step in range(num_steps):
            # Compute physics encoding
            phys_est = model.physics_enc(pc)
            
            # Apply self-reconstruction loss
            recon = model.physics_enc.decoder(phys_est)
            loss = F.mse_loss(recon, phys_est.detach())
            
            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Set model back to evaluation mode
    model.physics_enc.eval()
    
    # Disable gradients again
    for param in model.parameters():
        param.requires_grad = False
    
    return model 