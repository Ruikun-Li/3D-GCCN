# 3D-GCNN model
# The encoder and decoder of cnn can be replaced by any more powerful segmentation network

from .net_parts import *
from .gat_layers import *


# 定义Encoder_VGN_new
class Encoder3D(nn.Module):
    def __init__(self, channel_list=[1, 8, 16, 32, 64]):
        super(Encoder3D, self).__init__()
        self.input = DoubleConv3D(channel_list[0], channel_list[1])
        self.down1 = DownBlock3D(channel_list[1], channel_list[2])
        self.down2 = DownBlock3D(channel_list[2], channel_list[3])
        self.down3 = DownBlock3D(channel_list[3], channel_list[4])

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4


# 定义Decoder_VGN_new
class Decoder3D(nn.Module):
    def __init__(self, channel_list=[64, 32, 16, 8, 1]):
        super(Decoder3D, self).__init__()
        self.up3 = UpBlock3D(channel_list[0], channel_list[1], shortcut_ch=channel_list[1])
        self.up2 = UpBlock3D(channel_list[1], channel_list[2], shortcut_ch=channel_list[2])
        self.up1 = UpBlock3D(channel_list[2], channel_list[3], shortcut_ch=channel_list[3])
        self.output = nn.Conv3d(channel_list[3], channel_list[4], kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x_out = self.output(x)
        return torch.sigmoid(x_out), x


# Input:(N, in_features)(N, N), Output:(N, out_features)
class GAT3D(nn.Module):
    def __init__(self, n_feature, n_hidden, n_classes, dropout=0, alpha=0.2, n_heads=1):
        super(GAT3D, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(n_feature, n_hidden, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(n_hidden * n_heads, n_classes, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x_feat = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = self.out_att(x_feat, adj)
        return torch.sigmoid(x), x_feat  # (B, N, n_classes), (B, N, n_hidden * n_heads)


def trans_cnn_feature(cnn_feat, patch_nodes):  # (batch_size, n_feat, z, y, x)  (batch_size, n_feat, z, y, x)
    import numpy as np
    new_feature = np.zeros([patch_nodes.shape[0], patch_nodes.shape[1], cnn_feat.shape[1]])  # (batch_size, N, n_feat)
    new_feature = torch.from_numpy(new_feature.astype(np.float32))
    for batch_index in range(new_feature.shape[0]):
        for i in range(new_feature.shape[1]):
            new_feature[batch_index, i, :] = cnn_feat[batch_index, :, patch_nodes[batch_index, i, 0],
                                             patch_nodes[batch_index, i, 1], patch_nodes[batch_index, i, 2]]
    return new_feature

