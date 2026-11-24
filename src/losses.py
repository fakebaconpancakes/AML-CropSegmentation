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


def anomaly_suppression_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return our CAE adaptation of the paper's adaptive weighted MSE.

    Only masked AND valid pixels participate. RGB squared errors are averaged
    per pixel. For each image independently, raw weights are max(error) - error
    over selected pixels. Weights are detached and normalized to mean one.
    The final mean is over all selected pixels in the batch, as in baseline MSE.

    If an image's mean raw weight is at most eps (uniform/nearly uniform errors),
    use weights of one, recovering ordinary MSE for that image. Images without
    selected pixels contribute nothing; an entirely empty batch raises an error.
    Calculations use float32. No anomaly labels enter this loss.

    Per-batch updates, detachment, RGB reduction, selection masks, normalization,
    and the uniform-error fallback are explicit project adaptations, not a claim
    to reproduce every detail of Shikhar and Sobti (CVPR Workshop 2024).
    """
    if reconstruction.ndim != 4 or reconstruction.shape[1] != 3:
        raise ValueError("reconstruction must have shape (B, 3, H, W).")
    if target.shape != reconstruction.shape:
        raise ValueError("target must have the same shape as reconstruction.")
    if not reconstruction.is_floating_point() or not target.is_floating_point():
        raise ValueError("reconstruction and target must be floating-point tensors.")
    if not 0 < eps < float("inf"):
        raise ValueError("eps must be finite and positive.")

    batch_size, _, height, width = reconstruction.shape
    expected_shape = (batch_size, 1, height, width)
    for name, tensor in (("target", target), ("mask", mask), ("valid_mask", valid_mask)):
        if tensor.device != reconstruction.device:
            raise ValueError(f"{name} must be on the same device as reconstruction.")
    for name, tensor in (("mask", mask), ("valid_mask", valid_mask)):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError(f"{name} must contain only binary values 0 and 1.")

    selected = (mask.bool() & valid_mask.bool())[:, 0]
    if not selected.any():
        raise ValueError("No pixels are both masked and valid in this batch.")
    errors = (reconstruction.float() - target.float()).square().mean(dim=1)
    if not torch.isfinite(errors[selected]).all():
        raise ValueError("Selected reconstruction errors must be finite.")

    weighted_sums = []
    selected_count = 0
    for index in range(batch_size):
        pixel_errors = errors[index][selected[index]]
        if pixel_errors.numel() == 0:
            continue

        # No gradients through the weights, their maximum, or normalization.
        with torch.no_grad():
            raw_weights = pixel_errors.max() - pixel_errors
            mean_weight = raw_weights.mean()
            if mean_weight <= eps:
                weights = torch.ones_like(raw_weights)
            else:
                # Branching above avoids division by zero without changing the
                # requested mean-one normalization by adding eps to every mean.
                weights = raw_weights / mean_weight

        weighted_sums.append((weights * pixel_errors).sum())
        selected_count += pixel_errors.numel()

    return torch.stack(weighted_sums).sum() / selected_count
