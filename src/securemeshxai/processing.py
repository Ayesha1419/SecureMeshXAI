"""Leakage-controlled snapshot construction, train-only preprocessing and target construction."""
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from .constants import FEATURES, FLOW_FEATURES

@dataclass
class Snapshot:
    x: torch.Tensor             # (N,10,24) temporal features
    edge_index: torch.Tensor    # (2,E), directed reciprocal edges
    edge_attr: torch.Tensor     # (E,3)
    y: torch.Tensor             # (N,)
    trust_y: torch.Tensor       # (N,)
    active: torch.Tensor        # (N,), supervision only for observed source nodes
    node_ids: list
    timestamp: float
    capture: str
    def to(self,device):
        return Snapshot(*(v.to(device) if isinstance(v,torch.Tensor) else v for v in
                         (self.x,self.edge_index,self.edge_attr,self.y,self.trust_y,self.active,self.node_ids,self.timestamp,self.capture)))

BASE_FEATURES=['duration','orig_bytes','resp_bytes','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes','missed_bytes']

def raw_flow_features(df):
    """Twenty numeric protocol/state indicators before train-only scaling."""
    n=len(df)
    vals=np.column_stack([df[k].to_numpy(dtype=float) for k in BASE_FEATURES])
    by=vals[:,1]+vals[:,2]; packets=vals[:,3]+vals[:,4]
    # Mean interarrival is a flow approximation; window aggregation handles actual connection spacing.
    derived=np.column_stack([vals[:,1]/np.maximum(vals[:,2],1),by/np.maximum(packets,1),
                             packets/np.maximum(vals[:,0],1e-3),vals[:,0]/np.maximum(packets-1,1)])
    p=df.proto.str.lower().to_numpy()
    proto=np.column_stack([p==x for x in ('tcp','udp','icmp')]).astype(float)
    states=df.conn_state.str.upper().to_numpy()
    success=np.isin(states,['SF','S1','S2','S3'])
    reset=np.isin(states,['RSTO','RSTR','RSTOS0','RSTRH'])
    state=np.column_stack([success,states=='S0',states=='REJ',reset,
                           ~(success|(states=='S0')|(states=='REJ')|reset)]).astype(float)
    result=np.concatenate([vals,derived,proto,state],axis=1)
    result[:,:12]=np.log1p(np.maximum(result[:,:12],0))
    return np.nan_to_num(result,nan=0.,posinf=0.,neginf=0.).astype(np.float32)

class FeaturePipeline:
    def __init__(self,window_seconds=10,sequence_length=10,max_nodes=256,seed=42):
        self.window_seconds=window_seconds;self.sequence_length=sequence_length
        self.max_nodes=max_nodes;self.seed=seed
        self.scaler=MinMaxScaler(clip=True)
        self.iforest=IsolationForest(n_estimators=200,random_state=seed,n_jobs=-1,contamination='auto')
        self.anomaly_cutoff=None
        self.classes=None;self.external=set()
        self.fitted=False
    def fit(self,train,binary=True):
        if train.empty: raise ValueError('Training split is empty')
        flow=raw_flow_features(train)
        self.scaler.fit(flow[:,:12]); benign=flow[train.binary_label.to_numpy()==0]
        if len(benign)<3: raise ValueError('Need at least 3 benign training flows for trust anomaly model')
        self.iforest.fit(benign)
        # Higher anomaly score means more anomalous.
        self.anomaly_cutoff=float(np.percentile(-self.iforest.score_samples(benign),95))
        if binary:self.classes=['normal','attack']
        else:self.classes=['normal']+sorted(set(train.loc[train.binary_label==1,'class_label'])-{'normal'})
        if len(self.classes)<2: raise ValueError('At least two classes required for IDS training')
        # Endpoint grouping is trained on training records only.
        src=set(train.src.dropna().astype(str));dest=set(train.dst.dropna().astype(str))
        local=src & dest
        ext=pd.concat([train.src,train.dst]).astype(str).value_counts()
        self.external=set(ext.loc[~ext.index.isin(local)].head(50).index)
        self.fitted=True
        return self
    def _map_endpoints(self,frame):
        d=frame.copy()
        # Internal hosts are those repeatedly seen as originators in this partition; no labels used.
        src_counts=d.src.value_counts()
        internal=set(src_counts[src_counts>=2].index)
        def mapid(ip):
            ip=str(ip)
            if ip in internal or ip in self.external:return ip
            if ip.startswith(('224.','239.','255.')) or ip in ('0.0.0.0','::','ff02::1'):return '__EXTERNAL__'
            return '__EXTERNAL__'
        d['src_node']=d.src.map(mapid);d['dst_node']=d.dst.map(mapid)
        return d
    def transform(self,frame,binary=True,max_snapshots=None):
        if not self.fitted:raise RuntimeError('Fit pipeline on train split first')
        if frame.empty:return []
        d=self._map_endpoints(frame)
        raw=raw_flow_features(d)
        raw[:,:12]=self.scaler.transform(raw[:,:12])
        raw=np.clip(raw,0,1).astype(np.float32)
        d['_anomaly']=(-self.iforest.score_samples(raw_flow_features(d))>self.anomaly_cutoff).astype(float)
        d['_idx']=np.arange(len(d))
        result=[]
        for capture,g in d.groupby('capture',sort=True):
            times=g.ts.to_numpy(dtype=float);base=float(np.floor(times.min()/self.window_seconds)*self.window_seconds)
            g=g.copy();g['_window']=np.floor((g.ts-base)/self.window_seconds).astype('int64')
            histories=defaultdict(lambda:deque(maxlen=self.sequence_length))
            trust_previous=defaultdict(lambda:1.0)
            last_window_seen={}
            recent_rates=defaultdict(lambda:deque(maxlen=self.sequence_length))
            recent_anomalies=defaultdict(lambda:deque(maxlen=self.sequence_length))
            recent_success=defaultdict(lambda:deque(maxlen=self.sequence_length))
            # Iterate only event-bearing windows; zero padding represents missing observations.
            for window,w in g.groupby('_window',sort=True):
                node_groups=defaultdict(list)
                for r in w.itertuples():node_groups[r.src_node].append(r.Index);node_groups[r.dst_node].append(r.Index)
                if len(node_groups)>self.max_nodes:
                    ranked=sorted(node_groups,key=lambda n:len(node_groups[n]),reverse=True)[:self.max_nodes]
                    node_groups={n:node_groups[n] for n in ranked}
                nodes=sorted(node_groups); n=len(nodes);positions={v:i for i,v in enumerate(nodes)}
                if not n:continue
                window_vectors={};outputs=[];targets=[];active=[]
                for node in nodes:
                    indices=np.unique(node_groups[node]); v=raw[indices,:20].mean(axis=0)
                    node_rows=d.loc[indices]
                    originated=node_rows[node_rows.src_node==node]
                    # Ground truth applies to originator; responders receive no positive supervision.
                    positive=len(originated)>0
                    label=int(originated.binary_label.mean()>=.5) if positive else 0
                    if binary: yi=label
                    else:
                        cl=(originated.class_label.mode().iloc[0] if positive else 'normal')
                        yi=self.classes.index(cl) if cl in self.classes else -1
                    fraction=float(originated.binary_label.mean()) if positive else 0.
                    trust_previous[node]=.8*trust_previous[node]+.2*(1.-fraction)
                    recent_rates[node].append(float(len(indices)))
                    recent_anomalies[node].append(float(node_rows['_anomaly'].mean()))
                    success=float(node_rows.conn_state.str.upper().isin(['SF','S1','S2','S3']).mean())
                    recent_success[node].append(success)
                    cv=np.std(recent_rates[node])/max(float(np.mean(recent_rates[node])),1e-6)
                    stability=max(0.,1.-min(1.,cv))
                    persistence=float(np.mean(recent_anomalies[node]))
                    # Neighbourhood agreement is assigned after node-level vectors exist.
                    window_vectors[node]=(v,success,stability,persistence)
                    outputs.append(yi);targets.append(trust_previous[node]);active.append(positive and yi>=0)
                edge_data=defaultdict(list)
                for r in w.itertuples():
                    if r.src_node in positions and r.dst_node in positions and r.src_node!=r.dst_node:
                        key=tuple(sorted((r.src_node,r.dst_node)));edge_data[key].append(r.Index)
                neighbours=defaultdict(list)
                for a,b in edge_data:neighbours[a].append(b);neighbours[b].append(a)
                features=[]
                for node in nodes:
                    v,success,stab,persist=window_vectors[node]
                    vec=np.array([success,float(len(node_groups[node]))],dtype=np.float64)
                    neigh=neighbours[node]
                    if neigh:
                        mv=np.mean([[window_vectors[z][1],len(node_groups[z])] for z in neigh],axis=0)
                        agreement=float(np.dot(vec,mv)/(np.linalg.norm(vec)*np.linalg.norm(mv)+1e-8))
                    else:agreement=0.5
                    e=np.array([success,stab,np.clip(agreement,0,1),persist],dtype=np.float32)
                    z=np.concatenate([v,e])
                    if node in last_window_seen:
                        missing=min(self.sequence_length,int(window-last_window_seen[node]-1))
                        for _ in range(max(0,missing)):
                            histories[node].append(np.zeros(len(FEATURES),dtype=np.float32))
                    histories[node].append(z)
                    last_window_seen[node]=int(window)
                    seq=np.zeros((self.sequence_length,len(FEATURES)),dtype=np.float32)
                    h=list(histories[node]);seq[-len(h):]=h
                    features.append(seq)
                edges=[];attrs=[]
                for (a,b),idxs in edge_data.items():
                    rows=d.loc[idxs]; attr=[len(idxs),float((rows.orig_bytes+rows.resp_bytes).sum()),float(rows.duration.mean())]
                    edges.extend([[positions[a],positions[b]],[positions[b],positions[a]]]);attrs.extend([attr,attr])
                if edges:
                    edge_attr=np.log1p(np.maximum(np.array(attrs,dtype=np.float32),0))
                    edge_attr=edge_attr/(1+edge_attr)  # dimensionless, bounded, no validation fitting
                    edge_index=torch.tensor(edges,dtype=torch.long).T.contiguous()
                else:
                    edge_index=torch.empty((2,0),dtype=torch.long);edge_attr=np.zeros((0,3),dtype=np.float32)
                result.append(Snapshot(torch.tensor(np.stack(features)),edge_index,
                            torch.tensor(edge_attr,dtype=torch.float32),torch.tensor(outputs,dtype=torch.long),
                            torch.tensor(targets,dtype=torch.float32),torch.tensor(active,dtype=torch.bool),
                            nodes,base+int(window)*self.window_seconds,str(capture)))
                if max_snapshots and len(result)>=max_snapshots:return result
        return result
