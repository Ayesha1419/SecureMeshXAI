"""Run all manuscript seeds; training starts afresh per run."""
import argparse,subprocess,sys
from pathlib import Path
from securemeshxai.utils import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--source',default='ton');p.add_argument('--output',default='outputs/multiseed')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--split',default='stratified')
    a=p.parse_args();results=[]
    for seed in [42,123,2024,3407,7777]:
        command=[sys.executable,'-m','securemeshxai.cli','--data',*a.data,'--source',a.source,
                 '--seed',str(seed),'--epochs',str(a.epochs),'--split',a.split,
                 '--output',str(Path(a.output)/f'seed_{seed}')]
        subprocess.run(command,check=True);results.append(seed)
    write_json(Path(a.output)/'seeds.json',results)
if __name__=='__main__':main()
