"""Non-destructive, auditable gateway policy replay from previously recorded inference rows.
This is NOT live Contiki-NG actuation; the adapter emits recommended control actions.
"""
import argparse,csv,json
from pathlib import Path
from securemeshxai.policy import TrustMitigator

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',required=True)
    parser.add_argument('--output',default='outputs/mitigation_actions.json')
    a=parser.parse_args()
    rows=json.loads(Path(a.predictions).read_text())
    rows.sort(key=lambda r:(r.get('capture',''),r['timestamp']))
    engine=TrustMitigator();actions=[]
    current_capture=None
    for item in rows:
        if item.get('capture')!=current_capture:
            engine.reset();current_capture=item.get('capture')
        dec=engine.update(item['node'],item['attack_probability'],item['trust_pred'])
        actions.append(dict(capture=current_capture,timestamp=item['timestamp'],**vars(dec)))
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(actions,indent=2))
    print(f'{len(actions)} logged decisions to {path}')
if __name__=='__main__':main()
