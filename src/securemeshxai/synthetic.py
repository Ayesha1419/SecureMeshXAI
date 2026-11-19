"""Synthetic schema-compliant data for SMOKE TESTS ONLY, not research results."""
import numpy as np
import pandas as pd

def make_synthetic(rows=5000,seed=42,captures=6):
    rng=np.random.default_rng(seed);data=[]
    for ci in range(captures):
        count=rows//captures+(1 if ci<rows%captures else 0)
        for i in range(count):
            attack=bool(rng.random()<.4)
            src=f'10.0.{ci}.{rng.integers(1,16)}'
            dst=f'10.0.{ci}.{rng.integers(1,16)}' if rng.random()<.5 else f'172.16.0.{rng.integers(1,18)}'
            data.append({'ts':1700000000+ci*5000+i*2,'src_ip':src,'dst_ip':dst,
                         'duration':rng.uniform(.01,5)*(.4 if attack else 1),
                         'src_bytes':rng.exponential(500)*(2 if attack else 1),
                         'dst_bytes':rng.exponential(150),'src_pkts':rng.integers(1,15)*(3 if attack else 1),
                         'dst_pkts':rng.integers(1,15),'src_ip_bytes':rng.exponential(550),
                         'dst_ip_bytes':rng.exponential(160),'missed_bytes':0,
                         'proto':rng.choice(['tcp','udp','icmp']),
                         'conn_state':rng.choice(['SF','S0','REJ','RSTO']) if attack else 'SF',
                         'label':int(attack),'type':rng.choice(['ddos','dos','scanning']) if attack else 'normal',
                         'capture':f'capture_{ci}'})
    return pd.DataFrame(data)
