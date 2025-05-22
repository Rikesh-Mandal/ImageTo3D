import torch
import torch.nn as nn
from torchvision.models import resnet50


def encoder_resnet():
    """Create ResNet50 backbone that maintains spatial dimensions for feature extraction"""
    model = resnet50(weights='IMAGENET1K_V2')
    #print(model)
    # Remove the final layers (avgpool and fc)
    backbone = nn.Sequential(*list(model.children())[:-2])

    # Convert all weights to float32 explicitly
    return backbone.float()


def decoder():
    """Create depth prediction head with proper upsampling to match 384x384 output"""
    return nn.Sequential(
        # Initial convolution to reduce channel depth
        nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        # Upsample block 1 (12x12 -> 24x24)
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        # Upsample block 2 (24x24 -> 48x48)
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        # Upsample block 3 (48x48 -> 96x96)
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        # Upsample block 4 (96x96 -> 192x192)
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        # Final upsample to reach 384x384
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        # Final 1x1 convolution to get single channel output
        nn.Conv2d(32, 1, kernel_size=1)
    ).float()


def loss_function():
    """Create depth estimation loss function with size matching"""

    def depth_loss(pred, target):
        # Ensure prediction matches target size
        if pred.size() != target.size():
            #print("Size Mismatch, resizing!")
            pred = nn.functional.interpolate(
                pred,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        #print("Pred and Target size match")
        # L1 loss component(Mean Absolute Error)
        l1_loss = nn.L1Loss()(pred, target)

        '''
        In depth maps sudden jumps in pixel values might mean 
        - a depth discontinuity
        - or a wrong prediction
        '''
        # Gradient difference components
        grad_x_pred = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])  # all except last col - all except 1st col
        grad_x_target = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        grad_y_pred = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])  # all except last row - all except 1st row
        grad_y_target = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        grad_loss = (
                nn.L1Loss()(grad_x_pred, grad_x_target) +
                nn.L1Loss()(grad_y_pred, grad_y_target)
        )
        '''
        overall accuracy of depth production is added with
        0.1 * grad loss so the model is not penalized hard
        '''
        return l1_loss + 0.1 * grad_loss

    return depth_loss


def create_optimizer(backbone, head, learning_rate=1e-4):
    """Create Adam optimizer for both backbone and head"""
    params = list(backbone.parameters()) + list(head.parameters())
    return torch.optim.Adam(params, lr=learning_rate)


def save_model(backbone, head, filepath):
    """Save model weights to file"""
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'head_state_dict': head.state_dict()
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device):
    """Load model weights from file"""
    checkpoint = torch.load(filepath, map_location=device)

    # Create new model instances
    backbone = encoder_resnet()
    head = decoder()

    # Load weights
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])

    # Move to device
    backbone = backbone.to(device)
    head = head.to(device)

    return backbone, head
