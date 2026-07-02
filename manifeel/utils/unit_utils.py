import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvPoolingHead(nn.Module):
    def __init__(self, input_channels,
                 conv1_out_channels=128,
                 conv2_out_channels=256,
                 conv3_out_channels=512,
                 kernel_size=3, padding=1, use_se=False,
                 pool_output_size=(1, 1), output_dim=None, dropout=0.0):
        # Small CNN over a taxel force grid. The input is already per-channel min-max
        # normalized to ~[-1,1] by the dataset, so this net is deliberately NORM-FREE:
        # earlier versions applied a LayerNorm after every conv plus a final out_norm on
        # the embedding, which subtract the per-sample mean and rescale to unit variance
        # -- that erases the *magnitude* of the force field (how hard / which direction
        # the contact is), the most discriminative cue for insertion. AdaptiveAvgPool is
        # kept because averaging preserves scale (a harder press -> larger activations ->
        # larger pooled value); only mean/variance normalization strips magnitude.
        #
        # pool_output_size: spatial size AFTER adaptive pooling. (1,1) is a global average
        #   pool; on a small taxel grid prefer the full (H,W) or a modest grid to retain
        #   spatial contrast. output_dim: if set, flatten the pooled (C,ph,pw) map and
        #   Linear-project to it so spatial structure survives pooling.
        super(ConvPoolingHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=conv1_out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels,
                               out_channels=conv2_out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels,
                               out_channels=conv3_out_channels,
                               kernel_size=kernel_size,
                               padding=padding)

        self.pool = nn.AdaptiveAvgPool2d(pool_output_size)
        self.drop = nn.Dropout(dropout)
        if output_dim is not None:
            ph, pw = pool_output_size
            self.proj = nn.Linear(conv3_out_channels * ph * pw, output_dim)
        else:
            self.proj = None
        if use_se:
            self.se1 = SEBlock(conv1_out_channels)
            self.se2 = SEBlock(conv2_out_channels)
        self.use_se = use_se

    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        if self.use_se:
            x = self.se1(x)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        if self.use_se:
            x = self.se2(x)
        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop(x)
        if self.proj is not None:
            x = self.proj(x)
        return x
    
class MlpHead(nn.Module):
    def __init__(self, input_dim, hidden1_dim=512, hidden2_dim=512, output_dim=4, dropout_rate=0.0):
        super(MlpHead, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden2_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp_layers(x)


class TactileMlpHead(nn.Module):
    """Flatten a (B, C, H, W) taxel force grid and encode it with an MLP.

    Wraps MlpHead so the obs encoder can hand it the raw grid (same call signature
    as ConvPoolingHead). Like ConvPoolingHead it is norm-free, so the per-channel
    [-1,1] input scale -- i.e. contact magnitude -- flows through to the embedding.
    """
    def __init__(self, input_channels, tactile_ff_shape,
                 hidden_dim=512, output_dim=128, dropout=0.0):
        super(TactileMlpHead, self).__init__()
        c, h, w = tactile_ff_shape
        self.mlp = MlpHead(input_dim=c * h * w, hidden1_dim=hidden_dim,
                           hidden2_dim=hidden_dim, output_dim=output_dim,
                           dropout_rate=dropout)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.mlp(x)
