from tqdm import tqdm
import torch
import torch.nn as nn


def train_step(backbone, head, batch, loss_fn, optimizer, device):
    """Training step with dimension checks"""
    rgb, depth = batch
    rgb = rgb.float().to(device)
    depth = depth.float().to(device)

    optimizer.zero_grad()  # clearing old gradients before new backpropagation

    # Forward pass
    features = backbone(rgb)
    pred = head(features)

    # Ensure prediction matches target size
    if pred.shape[-2:] != depth.shape[-2:]:
        pred = nn.functional.interpolate(pred, size=depth.shape[-2:], mode='bilinear')

    # Calculate loss
    loss = loss_fn(pred, depth)

    # Backward pass / calculating gradients
    loss.backward()
    # Updating model parameters using the gradients
    optimizer.step()

    return loss.item()


def validate(backbone, head, val_data, loss_fn, device, steps):
    """Validation with proper feature extraction"""
    backbone.eval()
    head.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in tqdm(range(steps), desc="Validating"):
            rgb, depth = next(val_data)
            rgb = rgb.float().to(device)
            depth = depth.float().to(device)

            features = backbone(rgb)
            pred = head(features)
            loss = loss_fn(pred, depth)

            total_loss += loss.item()

    # setting the model back to training mode
    backbone.train()
    head.train()
    return total_loss / steps