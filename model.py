import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

    def reset_parameters(self):
        for m in self.modules():
            self.weights_init(m)
        self.bias.data.fill_(0.0)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        self.tanh = nn.Tanh()

        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.reset_parameters()

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = F.softmax(beta, dim=-1)
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.att.data, gain=1.414)


class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(P)])
        self.att = Attention(hidden_dim, attn_drop)
        self.reset_parameters()

    def forward(self, h, mps):
        embeds = []
        for i in range(self.P):
            x = self.node_level[i](h, mps[i])
            embeds.append(x)

        z_mp = self.att(embeds)
        # embeds = torch.cat(embeds, dim=-1)
        # print(embeds.shape)
        return z_mp

    def reset_parameters(self):
        for layer in self.node_level:
            layer.reset_parameters()
        self.att.reset_parameters()


class MGCN(nn.Module):
    def __init__(self, hidden_dim, num_class, feats_dim_list, feat_drop, attn_drop, num_meta_path):
        super(MGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.encoder = Mp_encoder(num_meta_path, hidden_dim, attn_drop)
        self.classifier = nn.Linear(hidden_dim, num_class)
        self.reset_parameters()

    def forward(self, x, mps):  # p a s
        h = F.elu(self.fc_list[0](x))
        z = F.elu(self.encoder(h, mps))
        pred = self.classifier(z)

        return pred

    def get_embed(self, x, mps):
        h = F.elu(self.fc_list[0](x))
        z = self.encoder(h, mps)

        return z.detach()

    def reset_parameters(self):
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

if __name__ == '__main__':
    m = MGCN(64, 3, [128], 0.5, 0.5, 2)

