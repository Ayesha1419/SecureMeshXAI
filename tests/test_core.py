import numpy as np
import torch
from securemeshxai.constants import FEATURES
from securemeshxai.synthetic import make_synthetic
from securemeshxai.datasets import canonicalize,split_records
from securemeshxai.processing import FeaturePipeline
from securemeshxai.model import SecureMeshXAI
from securemeshxai.policy import TrustMitigator
from securemeshxai.training import train,predict
from securemeshxai.metrics import evaluate

def small_data():
    df=canonicalize(make_synthetic(rows=600,captures=4))
    a,b,c=split_records(df,mode='stratified')
    pipe=FeaturePipeline(max_nodes=64).fit(a)
    return [pipe.transform(z,max_snapshots=5) for z in [a,b,c]]

def test_feature_dimensions():
    assert len(FEATURES)==24
    train_set,val,test=small_data()
    assert train_set[0].x.ndim==3
    assert train_set[0].x.shape[-2:]==(10,24)
    assert train_set[0].edge_attr.shape[-1]==3
    assert train_set[0].y.shape==train_set[0].trust_y.shape

def test_model_and_gradients():
    snap=small_data()[0][0]
    model=SecureMeshXAI(classes=2)
    output=model(snap.x,snap.edge_index,snap.edge_attr,attention=True)
    assert output['logits'].shape==(len(snap.node_ids),2)
    assert output['trust'].shape==(len(snap.node_ids),)
    assert len(output['attention'])==2
    assert ((output['trust']>=0)&(output['trust']<=1)).all()
    output['logits'].sum().backward()
    assert model.temporal.gru.weight_ih_l0.grad is not None

def test_training_smoke(tmp_path):
    tr,va,te=small_data()
    model=SecureMeshXAI(hidden=32,heads=4,classes=2)
    result=train(model,tr,va,epochs=2,patience=2,output=tmp_path)
    assert (tmp_path/'model.pt').exists()
    pred=predict(model,te)
    assert len(pred['y'])>0
    stats=evaluate(pred)
    assert 0<=stats['accuracy']<=1

def test_policy_three_strikes():
    policy=TrustMitigator()
    a=[policy.update('device1',.75,.3) for _ in range(3)]
    assert a[0].state=='mitigate'
    assert a[2].action=='isolate'
    assert policy.update('device2',.01,.99).state=='normal'
