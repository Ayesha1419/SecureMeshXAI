[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24373248.svg)](https://doi.org/10.5281/zenodo.24373248)

# SecureMeshXAI

**Executable research implementation** of the proposed SecureMeshXAI temporal–graph intrusion detection and trust management framework.

It implements a 2-layer GRU + additive attention, a 2-layer edge-aware 4-head GATv2-style encoder in pure PyTorch, a node-wise gated fusion, multitask IDS/trust heads, per-capture graph snapshots, benchmark label-derived trust proxy, trust/mitigation state machine, classical baselines, ablations, SHAP and LIME explainers, statistical tools, and an offline mitigation-policy replay. 


## Installation

Python 3.10+ required. For a CPU install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
pip install -e '.[test]'
```

For explanations and XGBoost:

```bash
pip install -e '.[all]'
```

CUDA availability depends on separately installing a PyTorch build compatible with your GPU/driver. PyTorch Geometric is **optional**, since graph attention is written in pure PyTorch.

## Confirm installation (synthetic test)

```bash
python -m pytest -q
python -m securemeshxai.cli --demo --epochs 2 --max-snapshots 30 --output outputs/demo
python simulation/replay.py --predictions outputs/demo/predictions.json --output outputs/demo/actions.json
```

## Real datasets

Download data using the original owners' terms:
- TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets
- IoT-23: https://www.stratosphereips.org/datasets-iot23

Use the **network Zeek connection** CSV/TSV/logs, not telemetry-only temperature or power tables. Required fields: timestamp (`ts`), originator (`src_ip` or `id.orig_h`), responder (`dst_ip` or `id.resp_h`) and labels (`label`, `type`, `detailed-label`). Other optional canonical fields are bytes, packets, connection status, protocol, etc. Exact source column mappings are in `src/securemeshxai/datasets.py`. For original Zeek logs the `#fields` header is used automatically.

```bash
# Binary benchmark (recommended first)
python -m securemeshxai.cli --data data/raw/ton_network.csv --source ton \
    --split stratified --epochs 100 --output outputs/ton_binary

# Chronological evaluation, with 5 minute purging at boundaries
python -m securemeshxai.cli --data data/raw/ton_network.csv --source ton \
    --split chronological --epochs 100 --output outputs/ton_chrono

# Multiclass (classes inferred from training partition)
python -m securemeshxai.cli --data data/raw/ton_network.csv --source ton \
    --multiclass --epochs 100 --output outputs/ton_multiclass

# IoT-23 capture-disjoint experiment
python -m securemeshxai.cli --data data/raw/iot23/*.log --source iot23 \
    --split scenario --epochs 100 --output outputs/iot23

# Zero-shot TON-trained binary model evaluated on IoT-23
python -m securemeshxai.cli --data data/raw/ton_network.csv --source ton \
    --external data/raw/iot23/*.log --epochs 100 --output outputs/transfer

# Comparators, targeted ablations, explanations
python -m securemeshxai.cli --data data/raw/ton_network.csv --source ton \
    --baselines --ablations --explain --epochs 50 --output outputs/full

# Five independent runs
python scripts/multi_seed.py --data data/raw/ton_network.csv --epochs 100
```

`--max-rows` (default 250,000) caps input for resource management. **Set to a value covering the complete dataset** for full-scale experiments. `--max-snapshots` is a development limit. Large datasets should be processed on a sufficiently provisioned gateway/workstation.

To evaluate a saved checkpoint on additional input:

```bash
python scripts/evaluate.py --checkpoint outputs/ton_binary/model.pt \
    --preprocessor outputs/ton_binary/preprocessor.joblib \
    --data data/raw/iot23/capture.log --source iot23 \
    --output outputs/independent_eval.json
```

### Outputs

Every successful experiment produces `model.pt`, `preprocessor.joblib`, `history.json`, `metrics.json`, `predictions.json`, `manifest.json`; optional `baselines.json`, `ablations.json`, `external_metrics.json`, `explanation.json`. `manifest.json` explicitly marks synthetic demonstrations.

## Tests and licensing

`pytest` validates shapes, gradients, data handling, policy transitions and an end-to-end training smoke test. A permissive MIT-style license applies to original code in this repository only; dataset and Contiki code retain their own terms.

## Citation

If you use SecureMeshXAI in your research, please cite this software using the following DOI:

**SecureMeshXAI v1.0.0**

Zenodo DOI: https://doi.org/10.5281/zenodo.24373248

### BibTeX

```bibtex
@software{securemeshxai_2026,
  author = {Ayesha1419},
  title = {SecureMeshXAI},
  year = {2026},
  version = {1.0.0},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.24373248},
  url = {https://doi.org/10.5281/zenodo.24373248}
}
```

