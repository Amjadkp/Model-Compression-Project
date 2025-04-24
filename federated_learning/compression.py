import torch
import numpy as np

def apply_random_mask(update, sparsity=0.25, seed=42):
    """Apply random mask to update (structured update)."""
    torch.manual_seed(seed)
    mask = torch.rand(update.shape) > sparsity  # Keep 25% of elements
    masked_update = update * mask
    return masked_update, mask

def apply_subsampling(update, fraction=0.25, seed=42):
    """Subsample update (sketched update)."""
    torch.manual_seed(seed)
    total_elements = update.numel()
    sample_size = int(total_elements * fraction)
    indices = torch.randperm(total_elements)[:sample_size]
    subsampled_update = torch.zeros_like(update)
    subsampled_update.view(-1)[indices] = update.view(-1)[indices] / fraction  # Scale for unbiased estimator
    return subsampled_update, indices

def apply_quantization(update, bits=2):
    """Quantize update to b bits (sketched update)."""
    h_min = update.min()
    h_max = update.max()
    if h_max == h_min:
        return update, (h_min, h_max)
    
    intervals = 2 ** bits
    step = (h_max - h_min) / intervals
    quantized = torch.zeros_like(update)
    for i in range(intervals):
        lower = h_min + i * step
        upper = h_min + (i + 1) * step
        mask = (update >= lower) & (update < upper)
        prob = (update[mask] - lower) / (upper - lower)
        quantized[mask] = lower * (1 - prob) + upper * prob
    return quantized, (h_min, h_max)

def apply_random_rotation(update, seed=42):
    """Apply structured random rotation (Hadamard transform)."""
    np.random.seed(seed)
    n = update.numel()
    # Use Hadamard matrix (simplified for demonstration)
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.float) / np.sqrt(2)  # Example for 2x2
    # In practice, use a larger Hadamard matrix or FFT-based rotation
    rotated = update.clone()  # Placeholder; implement actual rotation
    return rotated