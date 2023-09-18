import torch.nn as nn
from einops.layers.torch import Rearrange

from layers.Conv2dWithConstraint import Conv2dWithConstraint
from layers.LinearWithConstraint import LinearWithConstraint


class EegnetModel(nn.Module):
    def __init__(self, feature_extraction):
        super().__init__()

        self.feature_extraction = feature_extraction

    def forward(self, x):
        return self.feature_extraction(x)


class EegnetBlock(nn.Module):
    def __init__(self,
                 n_convs=[4, 2, 16],
                 temporal_conv_size=32, temporal_conv_padding='same',
                 separable_conv_size=16, separable_conv_padding='same',
                 polling_size=[4, 8], pooling_kind=['avg', 'avg'], dropout_rate=0.5,
                 n_channels=22, n_times=256, n_classes=4, filters=1):
        super().__init__()

        pooling_kind_dict = {'avg': nn.AvgPool2d, 'max': nn.MaxPool2d}
        n_features = n_times
        temporal_conv_size = n_times // 4 if temporal_conv_size == 'auto' else temporal_conv_size

        # Temporal Conv
        if filters > 1:
            rearrange = Rearrange('n f c t -> n f c t')
        else:
            rearrange = Rearrange('b c t -> b 1 c t')
        conv1 = nn.Conv2d(filters, n_convs[0], (1, temporal_conv_size), bias=False, padding=temporal_conv_padding)
        batch1 = nn.BatchNorm2d(n_convs[0])

        n_features = n_features - temporal_conv_size + 1 if temporal_conv_padding == 'valid' else n_features

        # Spatial Conv
        conv2 = Conv2dWithConstraint(n_convs[0], n_convs[0] * n_convs[1],
                                     (n_channels, 1), bias=False, groups=n_convs[0], max_norm=1)
        batch2 = nn.BatchNorm2d(n_convs[0] * n_convs[1])
        activation1 = nn.ReLU()
        pool1 = pooling_kind_dict[pooling_kind[0]]((1, polling_size[0]))
        dropout1 = nn.Dropout(dropout_rate)

        n_features = n_features // polling_size[0]

        # Separable Conv
        separable_conv_size = n_features // 4 if separable_conv_size == 'auto' else separable_conv_size
        conv3 = nn.Conv2d(n_convs[0] * n_convs[1], n_convs[0] * n_convs[1], (1, separable_conv_size), bias=False,
                          padding=separable_conv_padding, groups=n_convs[0] * n_convs[1])
        conv4 = nn.Conv2d(n_convs[0] * n_convs[1], n_convs[2], 1, bias=False)
        batch3 = nn.BatchNorm2d(n_convs[2])
        activation2 = nn.ReLU()
        pool2 = pooling_kind_dict[pooling_kind[1]]((1, polling_size[1]))
        dropout2 = nn.Dropout(dropout_rate)

        n_features = n_features - separable_conv_size + 1 if separable_conv_padding == 'valid' else n_features
        n_features = n_features // polling_size[1]

        n_features = n_convs[2] * n_features

        temporal = nn.Sequential(rearrange, conv1, batch1)
        spatial = nn.Sequential(conv2, batch2, activation1, pool1, dropout1)
        separable = nn.Sequential(conv3, conv4, batch3, activation2, pool2, dropout2)

        feature_extraction = nn.Sequential(temporal, spatial, separable)

        self.net = EegnetModel(feature_extraction)

    def forward(self, x):
        return self.net(x)
