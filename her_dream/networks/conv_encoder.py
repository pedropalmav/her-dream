import torch
import torch.nn as nn

from .rmsnorm_2d import RMSNorm2D


class ConvEncoder(nn.Module):
    def __init__(self, config, input_shape):
        super().__init__()
        act = getattr(torch.nn, config.act)
        h, w, input_ch = input_shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        self.kernel_size = int(config.kernel_size)
        in_dim = input_ch
        layers = []
        for i, depth in enumerate(self.depths):
            layers.append(
                nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=depth,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding="same",
                    bias=True,
                )
            )
            layers.append(nn.MaxPool2d(2, 2))
            if config.norm:
                layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth
            h, w = h // 2, w // 2

        self.out_dim = self.depths[-1] * h * w
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        """Encode image-like observations with a CNN."""
        # (B, T, H, W, C)
        obs = obs - 0.5
        # (B*T, H, W, C)
        x = obs.reshape(-1, *obs.shape[-3:])
        # (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # (B*T, C_feat, H_feat, W_feat)
        x = self.layers(x)
        # (B*T, C_feat*H_feat*W_feat)
        x = x.reshape(x.shape[0], -1)
        # (B, T, C_feat*H_feat*W_feat)
        return x.reshape(*obs.shape[:-3], x.shape[-1])
