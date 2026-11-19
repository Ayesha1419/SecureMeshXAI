import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from .utils import set_seed,write_json


def class_weights(snapshots,classes):
    counts=np.zeros(classes,dtype=np.float64)
    for s in snapshots:
        for y in s.y[s.active].numpy():
            if 0<=y<classes:counts[y]+=1
    weights=np.where(counts>0,1/np.sqrt(np.maximum(counts,1)),0)
    if weights.sum()==0:raise ValueError('No labeled node-windows')
    weights=weights/np.mean(weights[weights>0]);return torch.tensor(weights,dtype=torch.float32)

def compute_loss(out,s,weights,lam=.5,beta=.1):
    mask=s.active & (s.y>=0)
    if not mask.any():return None
    lids=F.cross_entropy(out['logits'][mask],s.y[mask],weight=weights)
    ltrust=F.mse_loss(out['trust'][mask],s.trust_y[mask])
    src,dst=s.edge_index
    if src.numel():
        # Avoid forcing trust equality across opposite-class neighbours.
        same=(s.y[src]==s.y[dst]) & s.active[src] & s.active[dst]
        spatial=((out['trust'][src[same]]-out['trust'][dst[same]])**2).mean() if same.any() else lids.new_zeros(())
    else:spatial=lids.new_zeros(())
    return lids+lam*ltrust+beta*spatial

@torch.no_grad()
def predict(model,snapshots,device='cpu'):
    model.eval();ys=[];preds=[];probs=[];trusted=[];targets=[];rows=[]
    for s in snapshots:
        sb=s.to(device);out=model(sb.x,sb.edge_index,sb.edge_attr)
        mask=sb.active & (sb.y>=0)
        prob=out['logits'].softmax(-1)
        for i in torch.nonzero(mask).flatten().tolist():
            ys.append(int(sb.y[i]));preds.append(int(prob[i].argmax()))
            probs.append(prob[i].detach().cpu().numpy())
            trusted.append(float(out['trust'][i]));targets.append(float(sb.trust_y[i]))
            rows.append({'capture':s.capture,'timestamp':s.timestamp,'node':s.node_ids[i],
                         'y_true':ys[-1],'y_pred':preds[-1],'trust_target':targets[-1],
                         'trust_pred':trusted[-1],'attack_probability':float(1-prob[i,0])})
    return {'y':np.array(ys),'pred':np.array(preds),'proba':np.array(probs),
            'trust':np.array(trusted),'trust_y':np.array(targets),'rows':rows}

def train(model,train_data,val_data,epochs=100,lr=.001,weight_decay=1e-5,
          patience=10,seed=42,device='cpu',output=None,lam=.5,beta=.1):
    if not train_data:raise ValueError('No training snapshots')
    set_seed(seed);model.to(device)
    weights=class_weights(train_data,model.ids[-1].out_features).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5)
    best=-float('inf');best_epoch=0;best_state=copy.deepcopy(model.state_dict());bad=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();total=0.;n=0
        for ix in np.random.default_rng(seed+epoch).permutation(len(train_data)):
            s=train_data[int(ix)].to(device)
            opt.zero_grad();out=model(s.x,s.edge_index,s.edge_attr)
            loss=compute_loss(out,s,weights,lam,beta)
            if loss is None:continue
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();total+=float(loss.detach());n+=1
        val=predict(model,val_data or train_data,device)
        score=float(f1_score(val['y'],val['pred'],average='macro',zero_division=0)) if len(val['y']) else 0.
        sched.step(score)
        history.append({'epoch':epoch,'train_loss':total/max(n,1),'val_macro_f1':score,'lr':opt.param_groups[0]['lr']})
        if score>best+1e-7:
            best=score;bad=0;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        else:bad+=1
        if bad>=patience:break
    model.load_state_dict(best_state)
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        torch.save({'model_state':best_state,'epoch':best_epoch,'val_macro_f1':best},output/'model.pt')
        write_json(output/'history.json',history)
    return {'best_epoch':best_epoch,'best_val_macro_f1':best,'history':history}
