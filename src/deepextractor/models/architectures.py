import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

logger = logging.getLogger(__name__)


# 1D models


class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET1D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv1D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv1D(feature * 2, feature))

        self.bottleneck = DoubleConv1D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class DnCNN1D(nn.Module):
    def __init__(self, depth=12, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN1D, self).__init__()
        padding = 1
        layers = []

        layers.append(
            nn.Conv1d(
                in_channels=image_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=image_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.orthogonal_(m.weight)
                logger.debug("init weight")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class Autoencoder1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(Autoencoder1D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv1D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv1D(feature, feature))

        self.bottleneck = DoubleConv1D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        for down in self.downs:
            x = down(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# 2D models


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512]):
        super(UNET2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv2D(feature * 2, feature))

        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class Autoencoder2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512]):
        super(Autoencoder2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv2D(feature, feature))

        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        original_size = x.size()[-2:]
        downs_outputs = []
        for down in self.downs:
            x = down(x)
            downs_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        downs_outputs = downs_outputs[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            target_size = downs_outputs[idx // 2].size()[-2:]
            if x.size()[-2:] != target_size:
                x = F.interpolate(x, size=target_size)
            x = self.ups[idx + 1](x)

        if x.size()[-2:] != original_size:
            x = F.interpolate(x, size=original_size)

        return self.final_conv(x)


class ModifiedAutoencoder2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256]):
        super(ModifiedAutoencoder2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(feature * 2, feature, kernel_size=3, stride=1, padding=1),
                )
            )

        self.bottleneck = nn.Conv2d(
            features[-1], features[-1] * 2, kernel_size=3, stride=1, padding=1
        )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        original_size = x.size()[-2:]
        downs_outputs = []
        for down in self.downs:
            x = down(x)
            downs_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        downs_outputs = downs_outputs[::-1]
        for idx, up in enumerate(self.ups):
            x = up(x)
            target_size = downs_outputs[idx].size()[-2:]
            if x.size()[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        if x.size()[-2:] != original_size:
            x = F.interpolate(x, size=original_size)

        return self.final_conv(x)
