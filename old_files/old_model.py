import torchvision
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet18_Weights

def build_model(num_classes: int, trainable_layers: int = 3, in_channels: int = 5):
    '''
    Builds a Faster R-CNN model with a ResNet-18 backbone.
    Args:
        num_classes (int): The number of classes for the detection task (including background).
        trainable_layers (int): The number of trainable layers in the backbone.
    Returns:
        model (torch.nn.Module): The constructed Faster R-CNN model.
    '''
    backbone = resnet_fpn_backbone(
        backbone_name='resnet18',
        weights=ResNet18_Weights.DEFAULT,
        trainable_layers=trainable_layers
    )

    if in_channels != 3:
        old_conv = backbone.body.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if in_channels > 3:
                rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
                for channel_idx in range(3, in_channels):
                    new_conv.weight[:, channel_idx:channel_idx + 1, :, :] = rgb_mean

        backbone.body.conv1 = new_conv

    if in_channels == 7:
        image_mean = [0.485, 0.456, 0.406, 0.0, 0.0, 0.0, 0.0] # Treba pozriet ale asi ok
        image_std = [0.229, 0.224, 0.225, 1.0, 1.0, 1.0, 1.0]
    elif in_channels == 5:
        image_mean = [0.485, 0.456, 0.406, 0.0, 0.0]
        image_std = [0.229, 0.224, 0.225, 1.0, 1.0]
    elif in_channels == 3:
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
    else:
        image_mean = [0.0] * in_channels
        image_std = [1.0] * in_channels

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=376,
        max_size=672,
        image_mean=image_mean,
        image_std=image_std,
    )
    #print(model)
    return model