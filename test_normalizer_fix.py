#!/usr/bin/env python3

import torch
from equibot.policies.utils.norm import Normalizer

def test_normalizer_shape_mismatch():
    """Test that the Normalizer can handle shape mismatches."""
    print("Testing Normalizer with shape mismatch...")
    
    # Create normalizer with stats for a larger tensor (14 dims)
    stats = {
        "min": torch.zeros(14),
        "max": torch.ones(14)
    }
    normalizer = Normalizer(stats)
    
    # Create a smaller tensor (4 dims)
    test_tensor = torch.randn(1, 1, 4)
    print(f"Input tensor shape: {test_tensor.shape}")
    
    try:
        # Try to normalize the smaller tensor
        normalized = normalizer.normalize(test_tensor)
        print(f"Normalized tensor shape: {normalized.shape}")
        print("✅ normalize() successful!")
        
        # Try to unnormalize the tensor
        unnormalized = normalizer.unnormalize(normalized)
        print(f"Unnormalized tensor shape: {unnormalized.shape}")
        print("✅ unnormalize() successful!")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_normalizer_shape_mismatch()
    if success:
        print("\nTest passed! The normalizer fix is working correctly.")
        exit(0)
    else:
        print("\nTest failed! The normalizer fix still has issues.")
        exit(1) 