"""CLI for real data, smoke tests, external transfer, explanations, and baseline studies."""
import argparse,joblib
from pathlib import Path
import torch
import pandas as pd
from .datasets import load_files,canonicalize,split_records
from .processing import FeaturePipeline
from .model import SecureMeshXAI
from .training import train,predict
from .metrics import evaluate
from .baselines import classical_baselines,run_neural_ablation
from .synthetic import make_synthetic
from .utils import set_seed,write_json,device_of
from .plots import save_plots

def pipeline_data(args,df,source):
    mode=args.split
    tr,va,te=split_records(df,mode=mode,source=source,seed=args.seed,
                            gap_seconds=args.gap_seconds if mode=='chronological' else 0)
    print(f'Raw split rows: train={len(tr)} val={len(va)} test={len(te)}')
    if any(x.empty for x in (tr,va,te)):
        raise ValueError('Empty partition. Use --split stratified or more timestamp/capture coverage.')
    processor=FeaturePipeline(seed=args.seed,max_nodes=args.max_nodes)
    processor.fit(tr,binary=not args.multiclass)
    dat=[processor.transform(x,binary=not args.multiclass,max_snapshots=args.max_snapshots) for x in (tr,va,te)]
    if not all(dat):raise ValueError('Missing graph snapshots; inspect timestamps or increase input rows')
    print('Graph snapshots:',[len(x) for x in dat],'Classes:',processor.classes)
    return processor,dat

def main(argv=None):
    parser=argparse.ArgumentParser(prog='securemeshxai')
    parser.add_argument('--data',nargs='+',help='TON_IoT or IoT-23 CSV, TSV, Zeek log or Parquet files')
    parser.add_argument('--source',choices=['ton','iot23'],default='ton')
    parser.add_argument('--split',choices=['chronological','stratified','scenario'],default='stratified')
    parser.add_argument('--gap-seconds',type=int,default=300)
    parser.add_argument('--output',default='outputs/experiment')
    parser.add_argument('--max-rows',type=int,default=250000)
    parser.add_argument('--max-snapshots',type=int,default=None)
    parser.add_argument('--max-nodes',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--multiclass',action='store_true')
    parser.add_argument('--demo',action='store_true',help='Generate synthetic SMOKE TEST data; not an experiment')
    parser.add_argument('--baselines',action='store_true',help='Evaluate classical classifiers')
    parser.add_argument('--ablations',action='store_true')
    parser.add_argument('--external',nargs='+',help='IoT-23 files for zero-shot TON_IoT binary transfer')
    parser.add_argument('--explain',action='store_true',help='Generate integrated-gradient, LIME and attention explanations')
    args=parser.parse_args(argv)
    set_seed(args.seed);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    dev=device_of(args.device)
    if args.demo:
        df=canonicalize(make_synthetic(rows=2400,seed=args.seed),'ton');source='ton'
        print('SYNTHETIC SMOKE TEST: metrics MUST NOT be used as paper results')
    else:
        if not args.data:parser.error('Supply --data PATH [PATH ...] or --demo')
        df=load_files(args.data,args.source,args.max_rows);source=args.source
    if args.external and args.multiclass:parser.error('External transfer requires binary classifier')
    processor,(tr,va,te)=pipeline_data(args,df,source)
    net=SecureMeshXAI(classes=len(processor.classes))
    fit=train(net,tr,va,epochs=args.epochs,seed=args.seed,device=dev,output=out)
    predictions=predict(net,te,dev);metrics=evaluate(predictions)
    write_json(out/'metrics.json',metrics);write_json(out/'predictions.json',predictions['rows'])
    joblib.dump(processor,out/'preprocessor.joblib')
    save_plots(predictions,out/'figures')
    write_json(out/'manifest.json',{'source':source,'synthetic_demo':args.demo,'seed':args.seed,
                     'split':args.split,'classes':processor.classes,'best_epoch':fit['best_epoch'],
                     'torch':torch.__version__,'source_files':args.data})
    print('TEST METRICS:',{k:round(v,4) for k,v in metrics.items() if isinstance(v,(float,int))})
    if args.baselines:
        write_json(out/'baselines.json',classical_baselines(tr,te,seed=args.seed))
    if args.ablations:
        write_json(out/'ablations.json',run_neural_ablation(tr,va,te,len(processor.classes),dev,args.seed,min(25,args.epochs)))
    if args.external:
        external=load_files(args.external,'iot23',args.max_rows)
        transformed=processor.transform(external,binary=True,max_snapshots=args.max_snapshots)
        ext=predict(net,transformed,dev)
        write_json(out/'external_metrics.json',evaluate(ext))
        write_json(out/'external_predictions.json',ext['rows'])
        print('Zero-shot external metrics saved')
    if args.explain:
        from .explain import NodeExplainer,trust_rationale
        sample=next((s for s in te if s.active.any()),None)
        idx=int(torch.nonzero(sample.active)[0]);expl=NodeExplainer(net,sample,idx,dev)
        report={'integrated_gradients':expl.integrated_gradients(16),
                'neighborhood':expl.neighbor_influence(),
                'trust_rationale':trust_rationale(sample.x[idx,-1].numpy())}
        try:report['lime']=expl.lime(num_samples=500)
        except ImportError:report['lime']='Optional dependency not installed'
        try:report['shap']=expl.shap()
        except ImportError:report['shap']='Optional dependency not installed'
        write_json(out/'explanation.json',report)
    print('Saved results to:',out.resolve())
    return 0
if __name__=='__main__':raise SystemExit(main())
