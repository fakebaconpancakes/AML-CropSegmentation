"""Masked reconstruction loss for valid agricultural image regions."""

import torch


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean RGB squared error over pixels that are masked AND valid.

    Args:
        reconstruction: Predicted floating-point RGB batch, shaped (B, 3, H, W).
        target: Original floating-point RGB batch with the same shape.
        mask: Binary tensor shaped (B, 1, H, W); 1 means masked from the input.
        valid_mask: Binary tensor shaped (B, 1, H, W); 1 means valid farmland.
            The caller must prepare this from the intersection of field bounds
            and validity masks. This function does not read dataset files.

    All tensors must be on the same device. Boolean or numeric binary masks
    are accepted. The mean covers selected pixels and all three RGB channels
    across the entire batch, rather than averaging individual image losses.
    An image with no selected pixels contributes nothing. If the entire batch
    has no selected pixels, a ValueError is raised instead of dividing by zero.
    """
    if reconstruction.ndim != 4 or reconstruction.shape[1] != 3:
        raise ValueError("reconstruction must have shape (B, 3, H, W).")
    if target.shape != reconstruction.shape:
        raise ValueError("target must have the same shape as reconstruction.")
    if not reconstruction.is_floating_point() or not target.is_floating_point():
        raise ValueError("reconstruction and target must be floating-point tensors.")

    batch_size, _, height, width = reconstruction.shape
    expected_mask_shape = (batch_size, 1, height, width)
    for name, tensor in (("target", target), ("mask", mask), ("valid_mask", valid_mask)):
        if tensor.device != reconstruction.device:
            raise ValueError(f"{name} must be on the same device as reconstruction.")
    for name, tensor in (("mask", mask), ("valid_mask", valid_mask)):
        if tensor.shape != expected_mask_shape:
            raise ValueError(f"{name} must have shape {expected_mask_shape}.")
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError(f"{name} must contain only binary values 0 and 1.")

    # Only the intersection of input masking and agricultural validity counts.
    selected_pixels = mask.bool() & valid_mask.bool()
    if not selected_pixels.any():
        raise ValueError("No pixels are both masked and valid in this batch.")

    squared_error = (reconstruction - target).square()
    # Expand across RGB so each selected pixel contributes three squared errors.
    selected_channels = selected_pixels.expand_as(squared_error)
    selected_errors = squared_error.masked_select(selected_channels)

    # mean() divides by (number of selected pixels * 3), preserving gradients.
    return selected_errors.mean()
