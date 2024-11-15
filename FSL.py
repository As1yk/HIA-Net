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
        if flag == 0:
            def supp_idxs(c):
                # FIXME when torch will support where as np
                return y.eq(c).nonzero()[:n_support].squeeze(1)

            classes = torch.unique(y)
            support_idxs = list(map(supp_idxs, classes))
            query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
            query_samples = x[query_idxs]  # (n_classes *n_query, dim)
            support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
            support_samples = support_samples.view(n_classes, n_support, -1)  # (n_classes, n_support, dim)

        else:
            data_src, data_tgt = x[0], x[1]
            label_src, label_tgt = y[0], y[1]
            classes = torch.unique(label_src)

            support_idxs = list(map(lambda c: label_src.eq(c).nonzero()[:n_support], classes))
            support_idxs = torch.stack(support_idxs).view(-1)
            support_samples = data_src[support_idxs].view(n_classes, n_support, -1)  # (n_classes, n_support, dim)

            query_idxs = list(map(lambda c: label_tgt.eq(c).nonzero()[:n_query], classes))
            query_idxs = torch.stack(query_idxs).view(-1)
            query_samples = data_tgt[query_idxs]  # (n_classes*n_query, dim)

        support_prototypes = support_samples.mean(1)  # (n_classes, dim)
        dists = self.euclidean_dist(query_samples, support_prototypes)
        return dists


