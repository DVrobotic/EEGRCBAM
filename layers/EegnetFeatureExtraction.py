import torch.nn as nn
from einops.layers.torch import Rearrange


class EegnetFeatureExtraction(nn.Module):
    def __init__(
            self,
            n_channels,
            kernel_length,
            F1,
            D,
            F2,
            pool1_stride,
            pool2_stride,
            dropout_rate
    ):
        super().__init__()

        # temporal
        self.rearrange = Rearrange('b c t -> b 1 c t')
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), bias=False, padding='same')
        self.batch1 = nn.BatchNorm2d(F1)

        # Spatial
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), bias=False, stride=(1, 1), groups=F1, padding="valid")
        self.batch2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, pool1_stride))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Separable Conv
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, kernel_length // 4), bias=False, padding='same', groups=F1 * D)
        self.conv4 = nn.Conv2d(F2, F2, 1, bias=False, groups=F2)
        self.batch3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, pool2_stride))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classifier
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Spectral
        out = self.rearrange(x)
        out = self.conv1(out)
        out = self.batch1(out)

        # Spatial
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.activation1(out)
        out = self.avgpool1(out)
        out = self.dropout1(out)

        # Temporal
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.batch3(out)
        out = self.activation2(out)
        out = self.avgpool2(out)
        out = self.dropout2(out)
        out = self.flatten(out)

        return out
