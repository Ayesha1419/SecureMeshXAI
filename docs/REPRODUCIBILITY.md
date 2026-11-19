# Reproducibility and manuscript-to-code mapping

| Manuscript module | Implementation | Qualification |
|---|---|---|
| Sec 4.3 dataset harmonization | datasets.py | Fallback for missing optional counters is zero |
| Sec 4.3 10-second graph snapshots | processing.py | IP communication graph; not physical RPL topology |
| Sec 4.4 24 features | constants.py, processing.py | Trust feature definitions adapted to available flow data |
| Sec 4.4 label-derived trust target | processing.py | Targets NEVER input; benchmark proxy only |
| Sec 4.5 GRU + GATv2 | model.py | Edge-aware pure-PyTorch GATv2-style message passing |
| Sec 4.5 multitask losses | training.py | Spatial consistency implemented; explicit temporal loss absent |
| Sec 4.6 XAI | explain.py | SHAP GradientExplainer, LIME, attention + edge deletion, rationale |
| Sec 4.7 mitigation | policy.py | Decision state machine; no live enforcement |
| Sec 4.8 train/validation/test | datasets.py | Stratified raw-flow or chronology/scenario split |
| Sec 5 metrics | metrics.py | Macro/micro metric semantics reported in JSON |
| Sec 5 baseline comparison | baselines.py | DT, RF, SVM, optional XGBoost; neural ablations |
| Sec 5 sensitivity | baselines.py | Selected architecture ablations only |
| Sec 5.10 routing simulation | simulation/replay.py | Offline only; missing Contiki patches/scenario setups |

The paper reports outcomes on unprovided exact artifacts. The released package is a validated implementation of its core design, **not an exact reproduction of every experimental table**. Never substitute synthetic demonstrations for published comparisons. No real benchmark data or model weights are bundled.

For high-confidence reproduction, record dataset checksums, source capture IDs, fixed class definitions, run manifests, environment versions and commit SHA. The script manifests retain seed, dataset mode, class labels and PyTorch version. Benchmarks should be rerun using all five paper seeds and leakage-controlled partitions.
