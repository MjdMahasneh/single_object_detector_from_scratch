import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale


class BBoxCNN(nn.Module):
    def __init__(self, backbone_name='resnet18', freeze_backbone=True):
        super(BBoxCNN, self).__init__()

        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == 'shufflenet':
            self.backbone = models.shufflenet_v2_x0_5(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.se = SEBlock(256)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            self.se,
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)



class BBoxCNNConvHead(nn.Module):
    def __init__(self, backbone_name='resnet18', freeze_backbone=True):
        super().__init__()

        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # keep up to last conv
            in_channels = 512
        else:
            raise NotImplementedError("Only resnet18 for now")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # compress to 1x1
            nn.Conv2d(256, 4, 1),  # 4 outputs: [x1, y1, x2, y2]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)   # [B, 512, H, W]
        x = self.head(x)       # [B, 4, 1, 1]
        return x.view(x.size(0), 4)  # flatten to [B, 4]
