import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, K = 64, dim = 128, alpha = 100.0,
                 normalize_input = True):
        """
        Args:
            K : int
                The number of clusters
            D : int
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
        # 把通道当特征维 把H*W当样本数
        self.conv = nn.Conv2d(dim, K, kernel_size=(1, 1), bias=True)
        # 特征中心同样优化
        self.centroids = nn.Parameter(torch.rand(K, dim))
        self._init_params()

    def _init_params(self):
        # W * x + b
        # 
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm( dim = 1)
        )

    def forward(self, x):
        # 使用论文中表示
        # H * W == N
        # x = B * D * H * W 
        B, D = x.shape[:2]
        
        if self.normalize_input:
            x = F.normalize(x, p = 2, dim = 1)  # across descriptor dim

        # soft-assignment，对每页上的K作为一个整体
        soft_assign = self.conv(x).view(B, self.K, -1)
        # B * K * N
        soft_assign = F.softmax(soft_assign, dim = 1)
        # B * D * N
        x_flatten = x.view(B, D, -1)
        
        # expand 
        #   新增加的维度将附在前面 扩大张量不需要分配新内存仅仅是新建一个张量的视图
        #   任意一个一维张量在不分配新内存情况下都可以扩展为任意的维度。
        #   传入 -1 则意味着维度扩大不涉及这个维度。
        # permute 调换顺序，相当于索引过去
        # 结合起来，就应该是直接看最后结果
        # H * W -> 3 * H * W
        # 那么 0 1 2 得到的都是H*W的第一个值
        # 交换位置后 H * W * 3
        # 最后一个位置才行
        
        # calculate residuals to each clusters
        # 每个 D 维 找到其对应第 K 个 D 维 的权重 在 K 维度中 
        # a: (B, D, N)  -> (B, K, D, N)
        # b: (K , D)  -> (K, D, N)    
        # residual: (B, K, D, N) 
        # K * D 代表 一个 D维度 描述子 和 所有中心的 差值 
        residual = x_flatten.expand(self.K, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # soft_assign: B * K * 1 *  N
        residual *= soft_assign.unsqueeze(2)
      
        # 所有样本加起来
        # B * K * D
        vlad = residual.sum(dim = -1)  
        # 针对描述子维度
        vlad = F.normalize(vlad, p = 2, dim = 2)  # intra-normalization 
        # B * (K*D)
        vlad = vlad.view(x.size(0), -1)  # flatten  
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
