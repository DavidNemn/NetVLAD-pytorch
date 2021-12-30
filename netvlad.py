import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters = 64, dim = 128, alpha = 100.0,
                 normalize_input = True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input

        
        # 1 * 1的卷积层相当于全连接
        #（batch,channel,height,width）
        # H * W 是 特征数目
        # channel 是 特征
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        # cin cout kw kh
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm( dim = 1)
        )

    def forward(self, x):
        #   H * W == 特征数目N
        # x = N * D * H * W 
        B, C = x.shape[:2]
        
        if self.normalize_input:
            x = F.normalize(x, p = 2, dim = 1)  # across descriptor dim

        # soft-assignment
        # 
        # (N, C, H, W) -> (N, num_clusters, H, W) -> (N, num_clusters, H * W)
        soft_assign = self.conv(x).view(B, self.num_clusters, -1)
        
        # (N, num_clusters, H * W)
        # N个矩阵
        # 矩阵 K * N  每一列代表描述子 跟 K 个类的权重
        soft_assign = F.softmax(soft_assign, dim = 1)

        
        x_flatten = x.view(B, C, -1) # (N, C, H, W) -> (N, C, H * W)
        
       
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # soft_assign: (N, num_clusters, H * W) -> (N, num_clusters, 1, H * W)
        residual *= soft_assign.unsqueeze(2)
      
        # (N, num_clusters, C, H * W) * (N, num_clusters, 1, H * W)
        vlad = residual.sum(dim = -1)  # (N, num_clusters, C, H * W) -> (N, num_clusters, C)

        vlad = F.normalize(vlad, p = 2, dim = 2)  # intra-normalization 
        vlad = vlad.view(x.size(0), -1)  # flatten   # flatten vald: (N, num_clusters, C) -> (N, num_clusters * C)
        vlad = F.normalize(vlad, p = 2, dim = 1)  # L2 normalize
        return vlad


class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        
        embedded_x = self.net_vlad(x)
        return embedded_x


class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)
