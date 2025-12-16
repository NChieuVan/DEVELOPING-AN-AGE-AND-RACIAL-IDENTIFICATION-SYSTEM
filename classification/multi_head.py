import torch
import torch.nn as nn
from torchvision import models

# =========================
# CBAM Attention
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        x_out = self.sigmoid(x_out)
        return x * x_out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# =========================
# MultiHeadClassifier with CBAM
# =========================
class MultiHeadClassifier(nn.Module):
    def __init__(self, num_age_classes, num_race_classes, backbone_name="mobilenet_v3_large", pretrained=True, dropout=0.3):
        super().__init__()

        # --------------------
        # Backbone
        # --------------------
        if backbone_name == "mobilenet_v2":
            backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
            self.backbone = backbone.features
            self.feature_dim = backbone.last_channel

        elif backbone_name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            self.backbone = backbone.features
            self.feature_dim = backbone.features[-1][0].out_channels

        elif backbone_name == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
            self.backbone = backbone.features
            self.feature_dim = backbone.features[-1][0].out_channels

        elif backbone_name == "resnet50":
            backbone = models.resnet18(weights="IMAGENET1K_V1")
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_dim = backbone.fc.in_features

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --------------------
        # CBAM Attention
        # --------------------
        self.att = CBAM(self.feature_dim)

        # --------------------
        # Neck: AvgPool + Dropout
        # --------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)

        # --------------------
        # Multi-task heads: Age + Race (MLP 1 tầng)
        # --------------------
        self.fc_age = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_age_classes)
        )

        self.fc_race = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_race_classes)
        )

    def forward(self, x):
        x = self.backbone(x)      # Feature map [B, C, H, W]
        x = self.att(x)           # CBAM attention
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        age_logits = self.fc_age(x)
        race_logits = self.fc_race(x)
        return age_logits, race_logits
if __name__ == "__main__":
    # Example usage: change backbone_name to test other models
    backbone_name = "mobilenet_v2"
    print(f"Testing backbone: {backbone_name}")
    model = MultiHeadClassifier(num_age_classes=7, num_race_classes=5, backbone_name=backbone_name)
    dummy_input = torch.randn(4, 3, 224, 224)
    age_logits, race_logits = model(dummy_input)
    print("Age logits shape:", age_logits.shape)
    print("Race logits shape:", race_logits.shape)
        