"""
Image encoder: ViT backbone (via timm) projected to shared embedding space.
Handles PIL images or file paths; applies standard ImageNet normalization.
"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        output_dim: int = 512,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        feature_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, output_dim),
        )

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_images(self, images: list) -> torch.Tensor:
        """Accept PIL images or file paths."""
        tensors = []
        for img in images:
            if isinstance(img, (str, bytes)):
                img = Image.open(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                raise TypeError(f"Expected PIL Image or path, got {type(img)}")
            tensors.append(self.preprocess(img))
        return torch.stack(tensors)

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device

        if not isinstance(images, torch.Tensor):
            images = self.preprocess_images(images)

        images = images.to(device)
        features = self.backbone(images)
        return self.projection(features)
