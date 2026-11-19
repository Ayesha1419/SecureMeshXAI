from collections import defaultdict
import numpy as np
from sklearn.metrics import (accuracy_score,precision_recall_fscore_support,roc_auc_score,
                             matthews_corrcoef,confusion_matrix,mean_absolute_error,mean_squared_error)

def trust_tier(x):
    return np.where(np.asarray(x)>=.7,2,np.where(np.asarray(x)>=.4,1,0))

def evaluate(pred):
    y=np.asarray(pred['y']);p=np.asarray(pred['pred']);proba=np.asarray(pred['proba'])
    if len(y)==0:raise ValueError('No labeled predictions')
    multi=proba.shape[1]>2
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average='macro' if multi else 'binary',
                     pos_label=1,zero_division=0)
    binary_truth=y!=0;binary_pred=p!=0
    cm=confusion_matrix(binary_truth,binary_pred,labels=[False,True]);tn,fp,fn,tp=cm.ravel()
    try:
        auc=roc_auc_score(y,proba[:,1] if not multi else proba,multi_class='ovr' if multi else 'raise',
                          average='macro',labels=list(range(proba.shape[1])) if multi else None)
    except ValueError:auc=float('nan')
    trust=np.asarray(pred['trust']);trust_y=np.asarray(pred['trust_y'])
    rows=pred.get('rows',[]);by_node=defaultdict(list)
    for r in rows:by_node[(r['capture'],r['node'])].append((r['timestamp'],r['trust_pred'],r['y_true']))
    changes=[]
    for seq in by_node.values():
        seq.sort()
        for a,b in zip(seq,seq[1:]):
            if b[2]==0:changes.append(abs(b[1]-a[1]))
    return {'n_samples':int(len(y)),'accuracy':float(accuracy_score(y,p)),
            'precision':float(precision),'recall':float(recall),'f1':float(f1),
            'macro_f1':float(precision_recall_fscore_support(y,p,average='macro',zero_division=0)[2]),
            'roc_auc':float(auc),'mcc':float(matthews_corrcoef(y,p)),
            'fpr':float(fp/max(fp+tn,1)),'fnr':float(fn/max(fn+tp,1)),
            'confusion_binary':cm.tolist(),'trust_tier_accuracy':float(np.mean(trust_tier(trust)==trust_tier(trust_y))),
            'trust_mae':float(mean_absolute_error(trust_y,trust)),
            'trust_rmse':float(np.sqrt(mean_squared_error(trust_y,trust))),
            'trust_stability_index':float(1.-np.mean(changes)) if changes else None}

def paired_statistics(rows_a,rows_b):
    """Input equal-length vectors of per-seed metric values; calculates paired t and Wilcoxon."""
    from scipy import stats
    a=np.asarray(rows_a,dtype=float);b=np.asarray(rows_b,dtype=float)
    if a.shape!=b.shape or len(a)<2:raise ValueError('Requires >=2 paired runs')
    diff=a-b;n=len(diff);mean=float(diff.mean());sd=float(diff.std(ddof=1))
    se=sd/np.sqrt(n);ci=stats.t.interval(.95,n-1,loc=mean,scale=se) if sd else (mean,mean)
    t,p=stats.ttest_rel(a,b)
    try:w_p=float(stats.wilcoxon(diff,method='exact').pvalue)
    except ValueError:w_p=1.
    return {'difference':mean,'ci95':list(ci),'t':float(t),'p':float(p),
            'wilcoxon_p':w_p,'cohens_dz':float(mean/sd) if sd else None}
