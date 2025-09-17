#!/usr/bin/env python3
"""Test script to verify the refactored experiments use identical datasets."""
import torch
from common_data import get_points, SEED

def test_identical_datasets():
    """Test that both experiments use identical datasets."""
    n, d = 1000, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get datasets twice
    xa1, xb1 = get_points(n, d, device, seed=SEED, cache=True)
    xa2, xb2 = get_points(n, d, device, seed=SEED, cache=True)
    
    # Verify they're identical
    assert torch.allclose(xa1, xa2), "xa datasets don't match!"
    assert torch.allclose(xb1, xb2), "xb datasets don't match!"
    
    print(f"✓ Datasets are identical for n={n}, d={d}")
    print(f"  xa shape: {xa1.shape}, device: {xa1.device}")
    print(f"  xb shape: {xb1.shape}, device: {xb1.device}")
    print(f"  xa range: [{xa1.min():.4f}, {xa1.max():.4f}]")
    print(f"  xb range: [{xb1.min():.4f}, {xb1.max():.4f}]")

if __name__ == "__main__":
    test_identical_datasets()
    print("✓ All tests passed!")