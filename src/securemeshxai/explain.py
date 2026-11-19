"""Model-faithful perturbation explanations with fixed snapshot graph context.
Optional SHAP and LIME wrappers work on the same target node and 24 input features.
"""
import numpy as np
import torch
from .constants import FEATURES

class NodeExplainer:
    def __init__(self,model,snapshot,node_index,device='cpu'):
        self.model=model.to(device).eval();self.snapshot=snapshot.to(device)
        self.node_index=int(node_index);self.device=device
        if self.node_index<0 or self.node_index>=len(snapshot.node_ids):raise IndexError(node_index)
        self.base=self.snapshot.x.detach().clone()
    @torch.no_grad()
    def predict(self,rows):
        rows=np.asarray(rows,dtype=np.float32).reshape(-1,len(FEATURES))
        values=[]
        for row in rows:
            x=self.base.clone();x[self.node_index,-1,:]=torch.tensor(row,device=self.device)
            out=self.model(x,self.snapshot.edge_index,self.snapshot.edge_attr)
            values.append(out['logits'][self.node_index].softmax(-1).cpu().numpy())
        return np.stack(values)
    def lime(self,num_samples=5000):
        try:from lime.lime_tabular import LimeTabularExplainer
        except ImportError as e:raise ImportError('Install securemeshxai[xai] for LIME') from e
        x=self.base[self.node_index,-1].cpu().numpy()
        background=np.clip(x[None,:]+np.random.default_rng(42).normal(0,.05,(200,len(x))),0,1)
        expl=LimeTabularExplainer(background,feature_names=FEATURES,class_names=[str(i) for i in range(self.model.ids[-1].out_features)],
              mode='classification',discretize_continuous=False,kernel_width=.75*np.sqrt(len(FEATURES)))
        explanation=expl.explain_instance(x,self.predict,num_features=10,num_samples=num_samples)
        label=int(self.predict(x[None,:]).argmax())
        return {'class':label,'features':explanation.as_list(label=label)}
    def integrated_gradients(self,steps=32):
        """Dependency-free local gradient attributions; not described as SHAP."""
        base=self.base.detach();x=base[self.node_index,-1].detach().clone();baseline=torch.zeros_like(x)
        grads=[]
        for t in np.linspace(0,1,steps):
            b=base.clone();b[self.node_index,-1]=baseline+float(t)*(x-baseline);b.requires_grad_(True)
            out=self.model(b,self.snapshot.edge_index,self.snapshot.edge_attr)
            score=out['logits'][self.node_index,1:].logsumexp(0)-out['logits'][self.node_index,0]
            grad=torch.autograd.grad(score,b)[0][self.node_index,-1]
            grads.append(grad.detach())
        vals=(x-baseline)*torch.stack(grads).mean(0)
        return dict(zip(FEATURES,vals.cpu().numpy().astype(float).tolist()))
    def shap(self,background=None):
        try:import shap
        except ImportError as e:raise ImportError('Install securemeshxai[xai] for SHAP') from e
        base=self.base;idx=self.node_index;model=self.model;edge=self.snapshot.edge_index;attr=self.snapshot.edge_attr
        class Local(torch.nn.Module):
            def forward(self,rows):
                preds=[]
                for row in rows:
                    x=base.clone();x[idx,-1]=row
                    preds.append(model(x,edge,attr)['logits'][idx].softmax(-1))
                return torch.stack(preds)
        inp=base[idx,-1].detach()
        if background is None:
            rng=np.random.default_rng(42)
            bg=np.clip(inp.cpu().numpy()[None,:]+rng.normal(0,.05,(200,len(FEATURES))),0,1)
        else:bg=np.asarray(background,dtype=np.float32)
        expl=shap.GradientExplainer(Local(),torch.tensor(bg,dtype=torch.float32,device=self.device))
        values=expl.shap_values(inp[None,:])
        return {'shap_values':np.asarray(values).tolist(),'features':FEATURES}
    @torch.no_grad()
    def neighbor_influence(self,masking_top_k=3):
        out=self.model(self.base,self.snapshot.edge_index,self.snapshot.edge_attr,attention=True)
        influence={}
        # Attention in GATv2 is a diagnostic, not by itself a causal attribution.
        for ed,att in out['attention']:
            src,dst=ed
            for j,score in zip(src[dst==self.node_index].tolist(),att[dst==self.node_index].mean(-1).tolist()):
                if j!=self.node_index:influence[j]=influence.get(j,0.)+score
        scale=sum(influence.values())
        if scale>0: influence={k:v/scale for k,v in influence.items()}
        ranked=sorted(influence,key=lambda z:influence[z],reverse=True)
        # Explicit edge masking preserves edge attributes exactly.
        original=float(1-out['logits'][self.node_index].softmax(-1)[0]);effects=[]
        for j in ranked[:masking_top_k]:
            e=self.snapshot.edge_index;keep=~(((e[0]==j)&(e[1]==self.node_index))|((e[1]==j)&(e[0]==self.node_index)))
            new=self.model(self.base,e[:,keep],self.snapshot.edge_attr[keep])
            p=float(1-new['logits'][self.node_index].softmax(-1)[0])
            effects.append({'neighbor':self.snapshot.node_ids[j],'attention':influence[j],
                            'delta_attack_probability':original-p})
        return {'node':self.snapshot.node_ids[self.node_index],'neighbors':effects}

def trust_rationale(feature_vector,benign_means=None):
    vals=np.asarray(feature_vector,dtype=float)[-4:]
    reference=np.asarray(benign_means if benign_means is not None else [.9,.9,.8,.1])
    deviations=np.abs(vals-reference);idx=int(np.argmax(deviations))
    return {'dominant_factor':FEATURES[20+idx], 'value':float(vals[idx]),
            'deviation_from_reference':float(vals[idx]-reference[idx])}
