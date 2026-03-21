import torch
import torch.nn.functional as F

# Note: We use fill_value = 0.5 (gray) instead of 0.0 (black) 
# because setting pixels to zero in normalized space can sometimes trigger 
# extreme activations, whereas 0.5 acts as a neutral "missing data" baseline.
def apply_random_patchdrop(
    x: torch.Tensor, 
    drop_ratio: float, 
    patch_size: int = 14, 
    fill_value: float = 0.5
) -> torch.Tensor:
    """
    Randomly drops patches from a batch of images by setting them to a fill value.
    
    Args:
        x: Input image tensor of shape [B, C, H, W] in [0, 1] range.
        drop_ratio: Fraction of patches to drop (0.0 to 1.0).
        patch_size: The size of the patches.
        fill_value: The value to fill dropped patches with (0.5 is gray).
    """
    if drop_ratio <= 0.0:
        return x

    B, C, H, W = x.shape
    grid_h, grid_w = H // patch_size, W // patch_size
    num_patches = grid_h * grid_w
    num_drop = int(num_patches * drop_ratio)

    if num_drop == 0:
        return x

    # Create a binary mask of shape [B, num_patches] (1 = keep, 0 = drop)
    mask = torch.ones((B, num_patches), device=x.device)
    
    for i in range(B):
        # Randomly select indices to drop for each image in the batch
        drop_indices = torch.randperm(num_patches, device=x.device)[:num_drop]
        mask[i, drop_indices] = 0.0 

    # Reshape mask to grid dimensions [B, 1, grid_h, grid_w]
    mask = mask.view(B, 1, grid_h, grid_w)
    
    # Upsample mask to the original image resolution [B, 1, H, W]
    mask = F.interpolate(mask, scale_factor=patch_size, mode='nearest')

    # Apply the mask to the image
    x_dropped = x * mask + fill_value * (1 - mask)
    
    return x_dropped