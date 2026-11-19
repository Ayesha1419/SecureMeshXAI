"""Stateful, non-destructive policy engine. Actual enforcement requires trusted actuator integration."""
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Decision:
    node: str
    attack_probability: float
    raw_trust: float
    smoothed_trust: float
    state: str
    action: str
    consecutive_mitigations: int

class TrustMitigator:
    def __init__(self,gamma=.8,delta=.3,attack_threshold=.7,trust_threshold=.4):
        self.gamma=gamma;self.delta=delta;self.attack_threshold=attack_threshold
        self.trust_threshold=trust_threshold;self.previous=defaultdict(lambda:1.)
        self.consecutive=defaultdict(int)
        self.audit=[]
    def update(self,node,attack_probability,head_trust):
        p=max(0.,min(1.,float(attack_probability)));h=max(0.,min(1.,float(head_trust)))
        instant=self.delta*(1-p)+(1-self.delta)*h
        current=self.gamma*self.previous[node]+(1-self.gamma)*instant
        self.previous[node]=current
        if p>=self.attack_threshold or current<=self.trust_threshold:
            self.consecutive[node]+=1
            if p>=.95 or self.consecutive[node]>=3:action='isolate'
            elif p>=.85:action='soft_quarantine'
            else:action='rate_limit_route_reweight'
            state='mitigate'
        elif current<.7 or p>=.5:
            self.consecutive[node]=0;state='monitor';action='enhanced_monitoring'
        else:
            self.consecutive[node]=0;state='normal';action='none'
        decision=Decision(str(node),p,h,current,state,action,self.consecutive[node]);self.audit.append(decision)
        return decision
    def reset(self):self.previous.clear();self.consecutive.clear();self.audit.clear()
