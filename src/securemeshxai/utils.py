import json, random
from pathlib import Path
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def encode(o):
        if isinstance(o, np.generic): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        raise TypeError(str(type(o)))
    path.write_text(json.dumps(obj,indent=2,default=encode),encoding='utf-8')

def device_of(request='auto'):
    return torch.device('cuda' if request=='auto' and torch.cuda.is_available() else ('cpu' if request=='auto' else request))
