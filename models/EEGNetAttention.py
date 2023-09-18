import torch
import torch.nn.functional as F
import torch.nn as nn
from layers.LinearWithConstraint import LinearWithConstraint
from layers.CBAM import CBAM
from layers.EegnetBlock import EegnetBlock
from modules.SklearnStructure import SklearnStructure


class EEGNnetAttentionModel(nn.Module):
    def __init__(
            self,
            classes=4,
            filters=1,
            res=[True, True],
            temporal_conv_size=64,
    ):
        super().__init__()

        self.eg_out = 16 * 8
        self.res = res

        self.eegnet1 = nn.Sequential(
            EegnetBlock(n_convs=[8, 2, 16], temporal_conv_size=temporal_conv_size, dropout_rate=0.5, filters=filters),
        )

        self.eg_norm = nn.LayerNorm([16, 1, 8])

        self.cbam1 = nn.Sequential(
            CBAM(gate_channels=16, reduction_ratio=2, kernel_size=2)
        )

        self.cbam_norm = nn.LayerNorm([16, 1, 8])

        if self.res[1]:
            self.head = nn.Sequential(
                nn.Flatten(),
                LinearWithConstraint(in_features=2 * self.eg_out, out_features=classes, max_norm=0.25)
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                LinearWithConstraint(in_features=self.eg_out, out_features=classes, max_norm=0.25)
            )

    def forward(self, x, targets):
        eg = self.eg_norm(self.eegnet1(x))
        if self.res[0]:
            cb = self.cbam_norm(self.cbam1(eg) + eg)
        else:
            cb = self.cbam_norm(self.cbam1(eg))

        if self.res[1]:
            out_cat = torch.cat([eg, cb], dim=-1)
        else:
            out_cat = cb

        logits = self.head(out_cat)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class EEGNnetAttention(SklearnStructure):
    def __init__(self, res=[True, True], temporal_conv_size=64):
        super().__init__()
        self.res = res
        self.temporal_conv_size = temporal_conv_size

    def fit(self, X, y, lr=0.0009, iterations=2000, batchsize=64, device='cuda', validation=0, track=None, verbose=False):
        self.model = EEGNnetAttentionModel(res=self.res, temporal_conv_size=self.temporal_conv_size)
        return self.train_model(X, y, batchsize=batchsize, lr=lr, iterations=iterations, device=device, track=track,
                                validation=validation)
# %%
