import torch
import torchvision.models as models
from torch import nn

def get_resnet_backbone():
    model = models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc
    return backbone

# Load model
backbone = get_resnet_backbone()

# Print only MAIN layers (no bottleneck details)
main_layers = list(backbone.named_children())
print("Modified ResNet50 Blocks (for 384×384 input):")
for name, module in main_layers:
    print(f"{name}: {module.__class__.__name__}")