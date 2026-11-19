"""Generate graphics only from actual model predictions, never pre-populated paper values."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,ConfusionMatrixDisplay,confusion_matrix

def save_plots(pred,output):
    target=Path(output);target.mkdir(parents=True,exist_ok=True)
    y=np.asarray(pred['y']);yhat=np.asarray(pred['pred']);proba=np.asarray(pred['proba'])
    if len(y)==0:return []
    paths=[]
    fig,ax=plt.subplots(figsize=(6,5))
    labels=list(range(proba.shape[1]))
    ConfusionMatrixDisplay(confusion_matrix(y,yhat,labels=labels),display_labels=labels).plot(ax=ax,cmap='Blues',colorbar=False)
    ax.set_title('Held-out IDS confusion matrix');fig.tight_layout()
    p=target/'confusion_matrix.png';fig.savefig(p,dpi=600);plt.close(fig);paths.append(str(p))
    if proba.shape[1]==2 and len(set(y))==2:
        fpr,tpr,_=roc_curve(y,proba[:,1]);fig,ax=plt.subplots(figsize=(6,5));ax.plot(fpr,tpr,label=f'AUC = {auc(fpr,tpr):.3f}')
        ax.plot([0,1],[0,1],'--',alpha=.6);ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate')
        ax.set_title('Held-out ROC curve');ax.legend();fig.tight_layout()
        p=target/'roc_curve.png';fig.savefig(p,dpi=600);plt.close(fig);paths.append(str(p))
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter(pred['trust_y'],pred['trust'],s=7,alpha=.5)
    ax.plot([0,1],[0,1],'--',alpha=.5);ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xlabel('Label-derived proxy trust target');ax.set_ylabel('Predicted trust')
    ax.set_title('Trust proxy estimation on held-out data');fig.tight_layout()
    p=target/'trust_scatter.png';fig.savefig(p,dpi=600);plt.close(fig);paths.append(str(p))
    return paths
