"""Common-protocol classical and neural baseline implementations."""
import numpy as np
import torch
from torch import nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from .model import SecureMeshXAI
from .training import train,predict
from .metrics import evaluate

def tabular_arrays(snapshots):
    x=[];y=[]
    for s in snapshots:
        mask=s.active & (s.y>=0)
        x.extend(s.x[mask,-1].numpy());y.extend(s.y[mask].numpy())
    return np.asarray(x),np.asarray(y,dtype=int)

def classical_baselines(train_set,test_set,seed=42):
    x,y=tabular_arrays(train_set);xt,yt=tabular_arrays(test_set)
    if len(xt)==0:raise ValueError('Test set lacks labeled nodes')
    models={'DecisionTree':DecisionTreeClassifier(random_state=seed,class_weight='balanced'),
            'RandomForest':RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1,class_weight='balanced'),
            'SVM':SVC(probability=True,random_state=seed,class_weight='balanced')}
    try:
        from xgboost import XGBClassifier
        models['XGBoost']=XGBClassifier(n_estimators=200,random_state=seed,n_jobs=4)
    except ImportError:pass
    result={}
    for name,model in models.items():
        model.fit(x,y);p=model.predict_proba(xt)
        classes=model.classes_;dense=np.zeros((len(xt),int(max(classes.max(),yt.max()))+1))
        dense[:,classes]=p
        labels=dense.argmax(1)
        from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
        result[name]={'accuracy':float(accuracy_score(yt,labels)),
                      'macro_f1':float(f1_score(yt,labels,average='macro',zero_division=0)),
                      'mcc':float(matthews_corrcoef(yt,labels))}
    return result

def run_neural_ablation(train_data,val_data,test_data,classes,device='cpu',seed=42,epochs=25):
    variants={'Full':{},'NoTemporal':{'use_temporal':False},'NoGraph':{'use_graph':False},
              'NoTrustEvidence':{'use_trust_features':False},'SumFusion':{'fusion':'sum'},
              'ConcatFusion':{'fusion':'concat'}}
    results={}
    for name,kw in variants.items():
        torch.manual_seed(seed)
        net=SecureMeshXAI(classes=classes,**kw)
        train(net,train_data,val_data,device=device,epochs=epochs,seed=seed)
        results[name]=evaluate(predict(net,test_data,device))
    return results
