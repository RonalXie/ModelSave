import torch
import torch.nn as nn
from layer import *
import numpy as np

class graphconv(nn.Module):
    def __init__(self,conv_channels, residual_channels, gcn_depth, dropout, propalpha):
        super(graphconv, self).__init__()
        self.gconv1=mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
        self.gconv2=mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
        #32 32 2 0.3 0.05
    def forward(self,x,adp):
        #x=torch.Size([64, 32, 207, 13])
        x = self.gconv1(x, adp) + self.gconv2(x, adp.transpose(1, 0))
        return x



class graphFpn(nn.Module):
    def __init__(self, conv_channels, residual_channels, gcn_depth, dropout, propalpha):
        super(graphFpn, self).__init__()
        self.ks=[0.9,0.8,0.7,0.6,0.5]
        self.gconv=graphconv(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
        self.unpool=Unpool()


    def forward(self,x,adj):
        input=x
        b, c, h, w = x.size()
        new_x = x.permute(2, 0, 1, 3)
        new_x = x.contiguous().view(h, -1)
        # for i in range(len(self.ks)):
        # self.Pool.append(Pool(self.ks[i],new_x.size()[1]).cuda())

        Pool1 = Pool(self.ks[0], new_x.size()[1])
        Pool2 = Pool(self.ks[1], new_x.size()[1])
        Pool3 = Pool(self.ks[2], new_x.size()[1])
        Pool4 = Pool(self.ks[3], new_x.size()[1])
        Pool5 = Pool(self.ks[4], new_x.size()[1])

        C1 = F.relu(self.gconv(x, adj))

        C1 = C1.permute(2, 0, 1, 3)
        C1 = C1.contiguous().view(C1.size()[0], -1)
        g1, new_hp1, idx1 = Pool1(adj, C1)
        new_h1 = new_hp1.contiguous().view(new_hp1.size()[0], b, c, -1)
        new_h1 = new_h1.permute(1, 2, 0, 3)
        # new_h1=new_h1.view(b,c,-1,w)

        # print(new_h1.size(),g1.size())
        C2 = F.relu(self.gconv(new_h1, g1))

        C2 = C2.permute(2, 0, 1, 3)
        C2 = C2.contiguous().view(C2.size()[0], -1)
        g2, new_hp2, idx2 = Pool2(g1, C2)
        new_h2 = new_hp2.contiguous().view(new_hp2.size()[0], b, c, -1)
        new_h2 = new_h2.permute(1, 2, 0, 3)

        C3 = F.relu(self.gconv(new_h2, g2))

        C3 = C3.permute(2, 0, 1, 3)
        C3 = C3.contiguous().view(C3.size()[0], -1)
        g3, new_hp3, idx3 = Pool3(g2, C3)
        new_h3 = new_hp3.contiguous().view(new_hp3.size()[0], b, c, -1)
        new_h3 = new_h3.permute(1, 2, 0, 3)

        C4 = F.relu(self.gconv(new_h3, g3))

        C4 = C4.permute(2, 0, 1, 3)
        C4 = C4.contiguous().view(C4.size()[0], -1)
        g4, new_hp4, idx4 = Pool4(g3, C4)
        new_h4 = new_hp4.contiguous().view(new_hp4.size()[0], b, c, -1)
        new_h4 = new_h4.permute(1, 2, 0, 3)

        C5 = F.relu(self.gconv(new_h4, g4))

        C5 = C5.permute(2, 0, 1, 3)
        C5 = C5.contiguous().view(C5.size()[0], -1)
        g5, new_hp5, idx5 = Pool5(g4, C5)
        new_h5 = new_hp5.contiguous().view(new_hp5.size()[0], b, c, -1)
        new_h5 = new_h5.permute(1, 2, 0, 3)

        # re_h=new_h5.permute(2,0,1,3)
        # re_h=re_h.contiguous().view(re_h.size()[0],-1)

        P5 = new_hp5
        # print("p5:",P5.size())
        #       print(g5[0].size(),P5.size())


        P4 = new_hp4 + self.unpool(g4, P5, idx5)
        P3 = new_hp3 + self.unpool(g3, P4, idx4)
        P2 = new_hp2 + self.unpool(g2, P3, idx3)
        P1 = new_hp1 + self.unpool(g1, P2, idx2)
        out = self.unpool(adj, P1, idx1)

        out = out.contiguous().view(h, b, c, -1)
        out = out.permute(1, 2, 0, 3)


        return out


class timeconv(nn.Module):
    def __init__(self,residual_channels, conv_channels, dilation_factor=1):
        super(timeconv, self).__init__()
        self.filter_convs=dilated_inception(residual_channels, conv_channels, dilation_factor=1)
        self.gate_convs=dilated_inception(residual_channels, conv_channels, dilation_factor=1)
        self.dropout=0.3

    def forward(self,x):
        filter = self.filter_convs(x)
        filter = torch.tanh(filter)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Pool(nn.Module):

    def __init__(self, k, in_dim, p=0.3):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1).cuda("cuda:0")
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)


if __name__ == '__main__':
    data=torch.rand(size=(64, 32, 207, 13),dtype=torch.float)
    adj=torch.rand(size=(207,207),dtype=torch.float)
    net=graphFpn(32,32,2,0.3,0.05)
    out=net(data,adj)
    print(out.size())


