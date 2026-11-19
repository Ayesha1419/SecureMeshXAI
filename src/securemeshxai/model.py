"""Native PyTorch edge-aware GATv2; runs without torch-geometric.
Self loops, four attention heads and residual layer norms included.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class EdgeGATv2(nn.Module):
    def __init__(self,dim=128,heads=4,edge_dim=3,dropout=.3):
        super().__init__()
        assert dim%heads==0
        self.heads=heads;self.head_dim=dim//heads;self.dropout=dropout
        self.lin_l=nn.Linear(dim,dim,bias=False);self.lin_r=nn.Linear(dim,dim,bias=False)
        self.lin_e=nn.Linear(edge_dim,dim,bias=False)
        self.att=nn.Parameter(torch.empty(heads,self.head_dim))
        self.bias=nn.Parameter(torch.zeros(dim));self.norm=nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.att)
    def forward(self,x,edge_index,edge_attr,return_attention=False):
        n=x.size(0);device=x.device
        loops=torch.arange(n,device=device)
        loops=torch.stack([loops,loops],dim=0)
        ed=torch.cat([edge_index,loops],dim=1)
        ea=torch.cat([edge_attr,x.new_zeros(n,edge_attr.size(-1))],dim=0)
        src,dst=ed
        l=self.lin_l(x).reshape(n,self.heads,self.head_dim)
        r=self.lin_r(x).reshape(n,self.heads,self.head_dim)
        e=self.lin_e(ea).reshape(-1,self.heads,self.head_dim)
        logits=(F.leaky_relu(l[src]+r[dst]+e,negative_slope=.2)*self.att).sum(-1)
        maxv=x.new_full((n,self.heads),float('-inf'))
        maxv.scatter_reduce_(0,dst[:,None].expand_as(logits),logits,reduce='amax',include_self=True)
        weights=(logits-maxv[dst]).exp()
        den=x.new_zeros((n,self.heads));den.index_add_(0,dst,weights)
        weights=weights/den[dst].clamp_min(1e-12)
        drop_weights=F.dropout(weights,p=self.dropout,training=self.training)
        messages=l[src]*drop_weights.unsqueeze(-1)
        out=x.new_zeros((n,self.heads,self.head_dim))
        out.index_add_(0,dst,messages)
        out=out.reshape(n,-1)+self.bias
        out=self.norm(F.elu(out)+x)
        if return_attention:return out,(ed,weights)
        return out,None

class TemporalEncoder(nn.Module):
    def __init__(self,input_dim=24,hidden=128,layers=2,dropout=.3):
        super().__init__()
        self.gru=nn.GRU(input_dim,hidden,num_layers=layers,batch_first=True,
                        dropout=dropout if layers>1 else 0)
        self.score=nn.Linear(hidden,1)
    def forward(self,x):
        h,_=self.gru(x)
        weights=torch.softmax(self.score(h).squeeze(-1),dim=1)
        return (weights.unsqueeze(-1)*h).sum(1)

class SecureMeshXAI(nn.Module):
    def __init__(self,input_dim=24,hidden=128,heads=4,dropout=.3,classes=2,
                 use_temporal=True,use_graph=True,use_trust_features=True,fusion='gate'):
        super().__init__()
        self.use_temporal=use_temporal;self.use_graph=use_graph
        self.use_trust_features=use_trust_features;self.fusion=fusion
        self.temporal=TemporalEncoder(input_dim,hidden,dropout=dropout)
        self.graph_in=nn.Linear(input_dim,hidden)
        self.gat1=EdgeGATv2(hidden,heads,dropout=dropout)
        self.gat2=EdgeGATv2(hidden,heads,dropout=dropout)
        self.gate=nn.Linear(2*hidden,1)
        self.concat=nn.Linear(2*hidden,hidden)
        self.ids=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,classes))
        self.trust=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,seq,edge_index,edge_attr,attention=False):
        if not self.use_trust_features:
            seq=torch.cat([seq[:,:,:20],torch.zeros_like(seq[:,:,20:])],dim=-1)
        ht=self.temporal(seq) if self.use_temporal else torch.zeros(seq.size(0),self.gate.in_features//2,device=seq.device)
        weights=[]
        if self.use_graph:
            hg=self.graph_in(seq[:,-1,:]);hg,a=self.gat1(hg,edge_index,edge_attr,attention)
            if attention:weights.append(a)
            hg,a=self.gat2(hg,edge_index,edge_attr,attention)
            if attention:weights.append(a)
        else:hg=torch.zeros_like(ht)
        if not self.use_graph:z=ht;gate=ht.new_ones((len(ht),1))
        elif not self.use_temporal:z=hg;gate=hg.new_zeros((len(hg),1))
        elif self.fusion=='sum':z=.5*(ht+hg);gate=ht.new_full((len(ht),1),.5)
        elif self.fusion=='concat':z=self.concat(torch.cat([ht,hg],-1));gate=ht.new_full((len(ht),1),.5)
        else:
            gate=torch.sigmoid(self.gate(torch.cat([ht,hg],dim=1)))
            z=gate*ht+(1-gate)*hg
        return {'logits':self.ids(z),'trust':self.trust(z).squeeze(-1),
                'gate':gate.squeeze(-1),'embeddings':z,'attention':weights}
