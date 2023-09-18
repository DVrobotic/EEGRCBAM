import torch
from torch import nn


class Conv3dWithConstraint(nn.Conv3d):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv3dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv3dWithConstraint, self).forward(x)