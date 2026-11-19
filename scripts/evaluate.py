"""Evaluate an existing checkpoint using a fitted training-only preprocessor."""
import argparse,joblib,torch
from securemeshxai.datasets import load_files
from securemeshxai.model import SecureMeshXAI
from securemeshxai.training import predict
from securemeshxai.metrics import evaluate
from securemeshxai.utils import write_json,device_of

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--preprocessor',required=True)
    p.add_argument('--data',nargs='+',required=True);p.add_argument('--source',choices=['ton','iot23'],default='ton')
    p.add_argument('--output',default='outputs/re_evaluation.json');p.add_argument('--device',default='auto')
    a=p.parse_args();processor=joblib.load(a.preprocessor)
    frame=load_files(a.data,a.source)
    if a.source=='iot23' and len(processor.classes)!=2:raise ValueError('Zero-shot evaluation requires binary checkpoint')
    snapshots=processor.transform(frame,binary=(len(processor.classes)==2))
    dev=device_of(a.device)
    model=SecureMeshXAI(classes=len(processor.classes)).to(dev)
    ckpt=torch.load(a.checkpoint,map_location=dev,weights_only=True)
    model.load_state_dict(ckpt['model_state']);result=evaluate(predict(model,snapshots,dev))
    write_json(a.output,result);print(result)
if __name__=='__main__':main()
