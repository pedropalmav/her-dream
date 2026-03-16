import torch
import torch.nn as nn
import distributions as dists
import re
from .conv_decoder import ConvDecoder
from .mlp_head import MLPHead
from functools import partial


class MultiDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shapes):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) == 3 and re.match(config.cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(config.mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        self.all_keys = list(self.mlp_shapes.keys()) + list(self.cnn_shapes.keys())

        # Unlike the encoder, each decoder is initialized independently.
        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                config.cnn,
                deter,
                flat_stoch,
                shape,
            )
            self._image_dist = partial(
                getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist
            )
        if self.mlp_shapes:
            shape = (sum(sum(x) for x in self.mlp_shapes.values()),)
            config.mlp.shape = shape
            self._mlp = MLPHead(config.mlp, deter + flat_stoch)
            self._mlp_dist = partial(
                getattr(dists, str(config.mlp_dist.name)), **config.mlp_dist
            )

    def forward(self, stoch, deter):
        """Decode latent states into observation distributions."""
        # (B, T, S, K), (B, T, D)
        dists = {}
        if self.cnn_shapes:
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            # (B, T, H, W, C_sum)
            outputs = self._cnn(stoch, deter)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            split_sizes = [v[0] for v in self.mlp_shapes.values()]
            # (B, T, S*K + D)
            feat = torch.cat([stoch.reshape(*deter.shape[:-1], -1), deter], -1)
            outputs = self._mlp(feat)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._mlp_dist(output)
                    for key, output in zip(self.mlp_shapes.keys(), outputs)
                }
            )
        return dists
