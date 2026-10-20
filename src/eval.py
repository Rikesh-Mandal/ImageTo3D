import numpy as np
import torch


def compute_metrics(pred, target):
    """Calculate depth estimation metrics"""
    '''
    removing the channel dimension and moving the tensor
    to cpu since numpy() operations are not supported on gpu
    '''
    pred = pred.squeeze(1).cpu().numpy()  # (B,H,W)
    target = target.squeeze(1).cpu().numpy()

    # Mask out invalid depths (where depth == 0)
    mask = target > 0

    rmse = np.sqrt(np.mean((pred[mask] - target[mask]) ** 2))  # Root Mean Squared Error
    abs_rel = np.mean(np.abs(pred[mask] - target[mask]) / target[mask])  # Absolute Relative Error

    return {'rmse': rmse, 'abs_rel': abs_rel}


def evaluate(backbone, head, test_data, device, steps):
    """Full model evaluation"""
    backbone.eval()
    head.eval()
    metrics = {'rmse': 0.0, 'abs_rel': 0.0}

    with torch.no_grad():
        for _ in range(steps):
            rgb, depth = next(test_data)
            rgb = rgb.float().to(device)
            depth = depth.float().to(device)

            features = backbone(rgb)
            pred = head(features)

            batch_metrics = compute_metrics(pred, depth)
            metrics['rmse'] += batch_metrics['rmse']
            metrics['abs_rel'] += batch_metrics['abs_rel']

    metrics['rmse'] /= steps
    metrics['abs_rel'] /= steps

    return metrics
