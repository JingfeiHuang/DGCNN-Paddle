
import os
import sys
import copy
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def knn(x, k):

    inner = -2*paddle.matmul(paddle.transpose(x=x, perm=[0,2,1]), x)
    xx = paddle.sum(x**2, axis=1, keepdim=True)
    a = paddle.transpose(xx, perm=[0,2,1])
    pairwise_distance = -xx - inner - a
 
    idx = paddle.topk(x=pairwise_distance, k=k, axis=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):

    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.reshape([batch_size, -1, num_points])
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base_pre = paddle.arange(0, batch_size)
    idx_base = idx_base_pre.reshape([-1,1,1])*num_points
    idx = idx + idx_base
    idx = idx.reshape([-1]).numpy()
 
    _, num_dims, _ = x.shape

    x = x.transpose(perm=[0,2,1])
    x_array = x.reshape([batch_size*num_points, -1]).detach().cpu().numpy()
    feature = paddle.to_tensor(x_array[idx, :])
    feature = feature.reshape([batch_size, num_points, k, num_dims])
    x = x.reshape([batch_size, num_points, 1, num_dims]).tile(repeat_times=[1, 1, k, 1])
    
    feature = paddle.concat((feature-x, x), axis=3).transpose(perm=[0, 3, 1, 2])
  
    return feature


class PointNet(nn.Layer):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1D(3, 64, kernel_size=1, bias_attr=False)
        self.conv2 = nn.Conv1D(64, 64, kernel_size=1, bias_attr=False)
        self.conv3 = nn.Conv1D(64, 64, kernel_size=1, bias_attr=False)
        self.conv4 = nn.Conv1D(64, 128, kernel_size=1, bias_attr=False)
        self.conv5 = nn.Conv1D(128, args.emb_dims, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(64)
        self.bn3 = nn.BatchNorm1D(64)
        self.bn4 = nn.BatchNorm1D(128)
        self.bn5 = nn.BatchNorm1D(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias_attr=False)
        self.bn6 = nn.BatchNorm1D(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Layer):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2D(64)
        self.bn2 = nn.BatchNorm2D(64)
        self.bn3 = nn.BatchNorm2D(128)
        self.bn4 = nn.BatchNorm2D(256)
        self.bn5 = nn.BatchNorm1D(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2D(6, 64, kernel_size=1, bias_attr=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2D(64*2, 64, kernel_size=1, bias_attr=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2D(64*2, 128, kernel_size=1, bias_attr=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2D(128*2, 256, kernel_size=1, bias_attr=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1D(512, args.emb_dims, kernel_size=1, bias_attr=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias_attr=False)
        self.bn6 = nn.BatchNorm1D(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1D(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(axis=-1, keepdim=False)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(axis=-1, keepdim=False)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(axis=-1, keepdim=False)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(axis=-1, keepdim=False)

        x = paddle.concat((x1, x2, x3, x4), axis=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).reshape(shape=[batch_size, -1])
        x2 = F.adaptive_avg_pool1d(x, 1).reshape(shape=[batch_size, -1])
        x = paddle.concat((x1, x2), axis=1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
