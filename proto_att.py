import torch.nn as nn
from torch.nn import functional as F
import torch


class ProtoNet(nn.Module):
    '''
    Prototypes of each class in support data
    '''

    def __init__(self):
        super(ProtoNet, self).__init__()

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x(queries): N x D (n_classes*n_query, dim)
        # y(prototypes): M x D (n_classes, dim)
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)  # (n_classes*n_query, n_classes, dim)
        y = y.unsqueeze(0).expand(n, m, d)  # (n_classes*n_query, n_classes, dim)

        return torch.pow(x - y, 2).sum(2)  # (n_classes*n_query, n_classes)

    def forward(self, x, y, n_classes, n_support=None, n_query=None, flag=0):
        if flag == 0:  # support and query all come from source/target
            def supp_idxs(c):
                """
                为每个类 c 提取前 n_support 个样本的索引

                假设 y = torch.tensor([0, 1, 2, 1, 2, 0, 1]) 且 n_support = 2，函数的作用如下：
                如果调用 supp_idxs(1)，那么：
                y.eq(1) 返回 [False, True, False, True, False, False, True]
                .nonzero() 返回 [1, 3, 6]
                [:n_support] 截取前 2 个，结果为 [1, 3]
                .squeeze(1) 后，得到 [1, 3]，表示标签为 1 的前两个样本的索引。
                """
                # FIXME when torch will support where as np
                return y.eq(c).nonzero()[:n_support].squeeze(1)

            classes = torch.unique(y)
            # 为每个类 c 提取前 n_support 个样本的索引
            support_idxs = list(map(supp_idxs, classes))
            query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
            query_samples = x[query_idxs]  # (n_classes *n_query, dim)
            support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
            support_samples = support_samples.view(n_classes, n_support, -1)  # (n_classes, n_support, dim)

        else:  # support and query come from source and target respectively
            # support来自source,query来自target
            data_src, data_tgt = x[0], x[1]
            label_src, label_tgt = y[0], y[1]
            classes = torch.unique(label_src)
            # 为每个类 c 提取前 n_support 个样本的索引
            support_idxs = list(map(lambda c: label_src.eq(c).nonzero()[:n_support], classes))
            support_idxs = torch.stack(support_idxs).view(-1)
            support_samples = data_src[support_idxs].view(n_classes, n_support, -1)  # (n_classes, n_support, dim)
            # 为每个类 c 提取前 n_query 个样本的索引
            query_idxs = list(map(lambda c: label_tgt.eq(c).nonzero()[:n_query], classes))
            query_idxs = torch.stack(query_idxs).view(-1)
            query_samples = data_tgt[query_idxs]  # (n_classes*n_query, dim)

        # 对每个类别取平均值作为原型
        support_prototypes = support_samples.mean(1)  # (n_classes, dim)
        # 返回每个query到原型的欧氏距离
        dists = self.euclidean_dist(query_samples, support_prototypes)
        return dists


