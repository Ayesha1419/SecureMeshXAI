"""TON_IoT/IoT-23 Zeek conn-log and Cooja CSV adapters.
The original labels are NEVER included in features.
"""
from pathlib import Path
import numpy as np
import pandas as pd

MAPPING = {
    'ts': ['ts','timestamp','time','Time','Timestamp'],
    'src': ['src_ip','src','id.orig_h','id_orig_h','source_ip'],
    'dst': ['dst_ip','dst','id.resp_h','id_resp_h','destination_ip'],
    'duration': ['duration','dur'],
    'orig_bytes': ['src_bytes','orig_bytes'], 'resp_bytes':['dst_bytes','resp_bytes'],
    'orig_pkts':['src_pkts','orig_pkts'], 'resp_pkts':['dst_pkts','resp_pkts'],
    'orig_ip_bytes':['src_ip_bytes','orig_ip_bytes'],
    'resp_ip_bytes':['dst_ip_bytes','resp_ip_bytes'],
    'missed_bytes':['missed_bytes'], 'proto':['proto','protocol'],
    'conn_state':['conn_state','state'],
    'label':['label','binary_label','attack','detailed-label','detailed_label'],
    'attack_type':['type','attack_type','detailed-label','detailed_label'],
    'capture':['capture','scenario','capture_id']
}

def read_table(path, max_rows=None):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    if path.suffix.lower()=='.parquet': return pd.read_parquet(path).head(max_rows)
    # Zeek #separator files, ordinary TSV and CSV are supported.
    with path.open('r',encoding='utf-8',errors='replace') as f:
        head=f.readline()
    if head.startswith('#separator') or head.startswith('#fields'):
        fields=None
        with path.open('r',encoding='utf-8',errors='replace') as f:
            for line in f:
                if line.startswith('#fields'):
                    fields=line.rstrip('\r\n').split('\t')[1:];break
        if fields is None: raise ValueError('Zeek log lacks #fields header')
        return pd.read_csv(path,sep='\t',comment='#',names=fields,nrows=max_rows,low_memory=False)
    sep='\t' if path.suffix.lower() in ('.tsv','.log') else ','
    return pd.read_csv(path,sep=sep,nrows=max_rows,low_memory=False)

def canonicalize(df, source='ton', capture=None):
    d=df.copy(); out=pd.DataFrame(index=d.index)
    for target,aliases in MAPPING.items():
        key=next((a for a in aliases if a in d.columns),None)
        if key is not None: out[target]=d[key]
    mandatory=['ts','src','dst']
    missing=[x for x in mandatory if x not in out]
    if missing: raise ValueError(f'Missing endpoint/time fields {missing}; provided: {list(df.columns)}')
    if 'label' not in out and 'attack_type' not in out:
        raise ValueError('No label or attack type field found (required for supervised training)')
    ts=pd.to_numeric(out['ts'],errors='coerce')
    if ts.notna().mean()<0.9:
        ts=pd.to_datetime(out['ts'],errors='coerce',utc=True).astype('int64') / 1e9
    out['ts']=ts
    out['src']=out['src'].astype('string'); out['dst']=out['dst'].astype('string')
    if 'capture' not in out: out['capture']=str(capture or 'capture0')
    out['capture']=out['capture'].fillna(str(capture or 'capture0')).astype(str)
    attack=out['attack_type'].astype('string').fillna('') if 'attack_type' in out else pd.Series('',index=out.index,dtype='string')
    if 'label' in out:
        label=out['label'].astype('string').str.strip().str.lower().fillna('')
    else: label=attack.str.lower()
    benign={'0','0.0','normal','benign','background','-'}
    if source=='iot23':
        # IoT-23 Zeek labels often have "-   Malicious" / "Benign" and detailed malicious strings.
        bad=label.str.contains('malicious|attack|c&c|botnet|ddos|scan|partofahorizontalportscan',case=False,regex=True)
        if 'label' not in out: bad=~label.isin(benign)
        else: bad=bad | (~label.isin(benign) & ~label.str.contains('benign|background|normal',case=False,regex=True))
    else:
        num=pd.to_numeric(label,errors='coerce')
        bad=np.where(num.notna(),num.fillna(0).to_numpy()>0,~label.isin(benign).to_numpy())
    out['binary_label']=np.asarray(bad,dtype=np.int64)
    type_clean=attack.str.strip().str.lower().str.replace(r'[^a-z0-9]+','_',regex=True)
    out['class_label']=np.where(out['binary_label'].eq(0),'normal',np.where(type_clean.isin(['','normal','benign']), 'attack',type_clean))
    for name in ['duration','orig_bytes','resp_bytes','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes','missed_bytes']:
        if name not in out: out[name]=0.0
        out[name]=pd.to_numeric(out[name].replace(['-','(empty)',''],np.nan),errors='coerce').fillna(0.0).clip(lower=0)
    for name in ['proto','conn_state']:
        if name not in out: out[name]='other'
        out[name]=out[name].astype(str).str.lower().str.strip()
    out=out.replace([np.inf,-np.inf],np.nan).dropna(subset=['ts','src','dst'])
    out=out[(out.src.str.len()>0)&(out.dst.str.len()>0)]
    return out.sort_values(['capture','ts'],kind='stable').reset_index(drop=True)

def load_files(paths,source='ton',max_rows=None):
    if isinstance(paths,(str,Path)): paths=[paths]
    frames=[]; remaining=max_rows
    for path in paths:
        path=Path(path)
        count=remaining if remaining else None
        df=canonicalize(read_table(path,count),source,capture=path.stem)
        frames.append(df)
        if remaining is not None:
            remaining-=len(df)
            if remaining<=0:break
    if not frames: raise ValueError('No input files')
    return pd.concat(frames,ignore_index=True)

def split_records(df,mode='chronological',source='ton',seed=42,gap_seconds=300):
    """Splits RAW rows before fitting preprocessing, to prevent train/test leakage.
    stratified approximates manuscript flow stratification; nodes/windows are reconstructed per split.
    """
    from sklearn.model_selection import train_test_split
    if mode=='scenario':
        captures=sorted(df.capture.unique())
        if len(captures)<3: raise ValueError('Scenario split requires >=3 distinct capture IDs')
        rng=np.random.default_rng(seed); captures=list(rng.permutation(captures))
        if len(captures)>=23: a,b=captures[:15],captures[15:18]
        else:
            i=max(1,int(round(len(captures)*.65))); j=max(i+1,int(round(len(captures)*.8)))
            a,b=captures[:i],captures[i:j]
        c=[x for x in captures if x not in a and x not in b]
        if not c: raise ValueError('Not enough captures for scenario split')
        return tuple(df[df.capture.isin(cs)].copy().reset_index(drop=True) for cs in (a,b,c))
    if mode=='stratified':
        strat=df.binary_label if df.binary_label.value_counts().min()>=3 else None
        tr,tmp=train_test_split(df,test_size=.2,random_state=seed,stratify=strat)
        strat2=tmp.binary_label if tmp.binary_label.value_counts().min()>=2 else None
        va,te=train_test_split(tmp,test_size=.5,random_state=seed,stratify=strat2)
        return tuple(x.sort_values(['capture','ts']).reset_index(drop=True) for x in (tr,va,te))
    if mode!='chronological':raise ValueError(f'Unknown split {mode}')
    df=df.sort_values('ts').reset_index(drop=True)
    a,b=df.ts.quantile([.7,.8]).to_numpy()
    return (df[df.ts<a-gap_seconds].reset_index(drop=True),
            df[(df.ts>=a+gap_seconds)&(df.ts<b-gap_seconds)].reset_index(drop=True),
            df[df.ts>=b+gap_seconds].reset_index(drop=True))
