from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn


class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


# current 1x1 conv in spconv2x has a bug. It will be removed after the bug is fixed
class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output


class UBlock(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output

# 博子自己加的一个密度感知模块，使用一个小的Inceptin模块，并使用稀疏形式实现
# class DensityAwareNet2(nn.Module):
#     def __init__(self, in_channels, out_channels, indice_key_id=2):
#         super(Inception, self).__init__()
#         self.input_inception_conv = spconv.SparseSequential(
#                 spconv.SubMConv3d(
#                 1, 1, kernel_size=3, padding=1, bias=False, indice_key='subm2'))
            
#         # 在此处定义一个局部密度感知模块，使用inception结构进行感知，使用稀疏卷积实现
#         self.conv = spconv.SparseSequential(
#             norm_fn(nPlanes[0]), nn.ReLU(),
#             spconv.SparseConv3d(
#                 1,
#                 1,
#                 kernel_size=3,
#                 stride=1,
#                 padding = 1,
#                 bias=False,
#                 indice_key='spconv{}'.format(indice_key_id)))

#         self.output_inception_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
#         self.output_a = nn.Conv1d(1, 1, 1)

#     def forward(self, x):
#         x = self.input_inception_conv(x)
#         x = self.conv(x)
#         x = self.output_inception_layer(x).features
#         x = output_a(x)
#         return x

class DensityAwareNet(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, indice_key_id=2):
        super(DensityAwareNet, self).__init__()
        self.conv = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.output_a = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.output_a(x)
        return x 

import functools
import torch.nn.init as init
norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
class DensityAwareSparseNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, norm_fn=norm_fn, indice_key_id=99):
        super(DensityAwareSparseNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = spconv.SparseSequential(
                norm_fn(self.in_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=0,
                    bias=False,
                    indice_key='spconv_selfAdd{}'.format(indice_key_id)))
        
        
        
    # 博子注：这里没有使用激活函数，有点失误
    def forward(self, x):
        x = self.conv1(x)
        #x = nn.ReLu(x)
        return x

