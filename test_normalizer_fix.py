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
    
    # Test case 1: Smaller tensor (4 dims)
    test_tensor1 = torch.randn(1, 1, 4)
    print(f"\nTest case 1: Input tensor shape: {test_tensor1.shape}")
    
    try:
        # Try to normalize the smaller tensor
        normalized1 = normalizer.normalize(test_tensor1)
        print(f"Normalized tensor shape: {normalized1.shape}")
        print("✅ normalize() successful!")
        
        # Try to unnormalize the tensor
        unnormalized1 = normalizer.unnormalize(normalized1)
        print(f"Unnormalized tensor shape: {unnormalized1.shape}")
        print("✅ unnormalize() successful!")
    except Exception as e:
        print(f"❌ Test case 1 failed: {e}")
        return False
        
    # Test case 2: Different tensor shape (batch dimension)
    test_tensor2 = torch.randn(8, 1, 4)
    print(f"\nTest case 2: Input tensor shape: {test_tensor2.shape}")
    
    try:
        # Try to normalize the tensor with batch dimension
        normalized2 = normalizer.normalize(test_tensor2)
        print(f"Normalized tensor shape: {normalized2.shape}")
        print("✅ normalize() successful!")
        
        # Try to unnormalize the tensor
        unnormalized2 = normalizer.unnormalize(normalized2)
        print(f"Unnormalized tensor shape: {unnormalized2.shape}")
        print("✅ unnormalize() successful!")
    except Exception as e:
        print(f"❌ Test case 2 failed: {e}")
        return False
    
    # Test case 3: Try with a tensor that has exactly the right dimensions
    test_tensor3 = torch.randn(1, 1, 14)
    print(f"\nTest case 3: Input tensor shape: {test_tensor3.shape}")
    
    try:
        # Try to normalize the exact-sized tensor
        normalized3 = normalizer.normalize(test_tensor3)
        print(f"Normalized tensor shape: {normalized3.shape}")
        print("✅ normalize() successful!")
        
        # Try to unnormalize the tensor
        unnormalized3 = normalizer.unnormalize(normalized3)
        print(f"Unnormalized tensor shape: {unnormalized3.shape}")
        print("✅ unnormalize() successful!")
    except Exception as e:
        print(f"❌ Test case 3 failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = test_normalizer_shape_mismatch()
    if success:
        print("\nAll tests passed! The normalizer fix is working correctly.")
        exit(0)
    else:
        print("\nTests failed! The normalizer fix still has issues.")
        exit(1) 