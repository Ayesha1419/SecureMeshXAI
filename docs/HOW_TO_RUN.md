# Running SecureMeshXAI

## 1. Install

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
pip install -e '.[test]'
```

## 2. Verify the installation

```bash
pytest -q
python -m securemeshxai.cli --demo --epochs 2 --max-snapshots 12 --output outputs/demo
```

Synthetic smoke-test scores do not represent research evidence.

## 3. Download actual research datasets

Obtain Zeek connection logs from the official TON_IoT and IoT-23 dataset pages. For full evaluation do not use synthetic sample CSVs in `examples/`. If raw input has no originator/responder endpoint or valid labels, supervised training will reject it. `--max-rows` defaults to 250000, so increase as appropriate.

## 4. Train binary IDS on real TON_IoT

```bash
python -m securemeshxai.cli --data /path/to/TON_IoT_network.csv --source ton --epochs 100 --split stratified --output outputs/ton
```

## 5. Evaluate cross-dataset transfer

```bash
python -m securemeshxai.cli --data /path/to/TON_IoT_network.csv --source ton --external /path/to/iot23_capture.log --epochs 100 --output outputs/transfer
```

## 6. Run classical comparisons and ablations

```bash
python -m securemeshxai.cli --data /path/to/TON_IoT_network.csv --source ton --baselines --ablations --epochs 25 --output outputs/analysis
```

## 7. Explain predictions

```bash
pip install -e '.[xai]'
python -m securemeshxai.cli --data /path/to/TON_IoT_network.csv --source ton --explain --epochs 100 --output outputs/xai
```

## 8. Replay trust policy and mitigation recommendations

```bash
python simulation/replay.py --predictions outputs/ton/predictions.json --output outputs/ton/actions.json
```

This is decision replay only. Routing changes require Contiki-NG/RPL firmware and a separate reviewed actuator interface.
