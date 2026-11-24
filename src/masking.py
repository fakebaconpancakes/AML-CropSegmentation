"""Random square-patch masking for 128 x 128 RGB image batches."""

import torch


def random_patch_mask(
    images: torch.Tensor,
    patch_size: int = 8,
    mask_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace randomly selected patches with zeros without changing the input.

    Args:
        images: Image batch shaped (B, 3, 128, 128).
        patch_size: Positive integer dividing both image dimensions evenly.
        mask_ratio: Fraction of patches to mask, between 0 and 1 inclusive.

    Returns:
        Masked images with the input shape, and a binary mask shaped
        (B, 1, 128, 128). A value of 1 means masked; 0 means visible.
        Both outputs use the input's device and dtype.

    Each image gets an independent random selection without replacement.
    The number of masked patches is int(mask_ratio * total_patches), rounded
    down if necessary. Randomness uses PyTorch's current random generator;
    torch.manual_seed can be used for reproducible results.
    """
    if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
        raise ValueError("Expected input shape (batch_size, 3, 128, 128).")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer.")
    if not 0 <= mask_ratio <= 1:
        raise ValueError("mask_ratio must be between 0 and 1 inclusive.")

    batch_size, _, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must divide image height ({height}) "
            f"and width ({width}) evenly."
        )

    grid_height = height // patch_size
    grid_width = width // patch_size
    total_patches = grid_height * grid_width
    masked_count = int(mask_ratio * total_patches)

    # Start with visible patches, then choose a new random subset for each image.
    patch_mask = torch.zeros(
        (batch_size, total_patches), device=images.device, dtype=images.dtype
    )
    for index in range(batch_size):
        selected = torch.randperm(total_patches, device=images.device)[:masked_count]
        patch_mask[index, selected] = 1

    # Expand each grid value into a square patch of identical pixel values.
    mask = patch_mask.reshape(batch_size, 1, grid_height, grid_width)
    mask = mask.repeat_interleave(patch_size, dim=2)
    mask = mask.repeat_interleave(patch_size, dim=3)

    # Broadcast the single mask channel across RGB. masked_fill returns a new tensor.
    masked_images = images.masked_fill(mask.bool(), 0)
    return masked_images, mask
