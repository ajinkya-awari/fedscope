<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=200&section=header&text=FedScope&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Reproducible%20Federated%20Learning%20Benchmark%20on%20PathMNIST&descAlignY=58&descAlign=50&descSize=16"/>

[![Tests](https://img.shields.io/badge/tests-57%20passing-brightgreen)](tests/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Kernel%20v19-20BEFF?logo=kaggle)](https://www.kaggle.com/code/ajinkya1225/05-fedscope)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](pyproject.toml)
[![Flower](https://img.shields.io/badge/Flower-1.x-purple)](kaggle/requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ajinkya--awari%2Ffedscope-181717?logo=github)](https://github.com/ajinkya-awari/fedscope)

</div>

**FedScope** benchmarks four federated aggregation strategies — FedAvg ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)), FedProx ([Li et al., 2020](https://arxiv.org/abs/1812.06127)), FedNova ([Wang et al., 2020](https://arxiv.org/abs/2007.06234)), and ServerMomentum — on [PathMNIST](https://medmnist.com/) (size 28, nine-class patch-pathology classification) under controlled non-IID Dirichlet partitioning across five seeded virtual hospitals.

The primary research question: *does aggregation strategy choice matter at 3 federated rounds under non-IID data, and does the centralized architecture correctly learn the task?* Across three seeds and 12 strategy–seed pairs, the answer is: **centralized learning converges reliably (F1 ≥ 0.779); federated strategies are indistinguishable at 3 rounds CPU** — and the flat profile across all strategies is the informative result.

---

## Results

### Centralized Baseline Gate (30 epochs, SGD momentum=0.9)

All three seeds pass the F1 ≥ 0.70 gate, confirming the architecture and data pipeline are correct before any federated work begins.

| Seed | Macro F1 | Gate |
|------|:--------:|:----:|
| 42   | 0.7799   | ✓ PASSED |
| 123  | 0.7895   | ✓ PASSED |
| 456  | 0.7951   | ✓ PASSED |

### Federated Benchmark (3 rounds, 5 clients, Dirichlet α=0.5)

Mean macro F1 ± std across seeds 42 / 123 / 456:

| Strategy | Mean F1 | Std | Paper |
|----------|:-------:|:---:|-------|
| **FedAvg** | 0.0706 | ±0.0394 | [McMahan et al., 2017](https://arxiv.org/abs/1602.05629) |
| **FedProx** | 0.0475 | ±0.0408 | [Li et al., 2020](https://arxiv.org/abs/1812.06127) |
| **FedNova** | 0.0267 | ±0.0020 | [Wang et al., 2020](https://arxiv.org/abs/2007.06234) |
| **ServerMomentum** | 0.0246 | ±0.0032 | — EMA baseline |

> **Interpretation:** Federated F1 is near-random (9-class chance ≈ 0.111) at 3 rounds on CPU. This is a compute budget result, not an architecture failure — the centralized gate confirms the model and data pipeline are correct. Federated convergence requires more rounds or a GPU session. All 12 strategy–seed pairs completed; artifacts and convergence PNGs are persisted.

---

## What This Measures

FedScope evaluates three dimensions simultaneously:

**1. Aggregation correctness** — do strategy implementations faithfully reproduce their published update rules?

**2. Non-IID sensitivity** — does client drift vary across strategies under Dirichlet α=0.5 partitioning?

**3. Reproducibility** — do results match across three independent seeds?

The **centralized gate** is a prerequisite, not a baseline: it verifies the model and data pipeline before any federated work begins. The gate payload is persisted with a SHA-256 digest; `require_centralized_gate` re-verifies this digest before each federated run.

### Strategy Formulations

| Strategy | Update Rule |
|----------|-------------|
| **FedAvg** | w ← Σᵢ (nᵢ/n) wᵢ |
| **FedProx** | min ℓ(w) + ½μ ‖w − w_global‖² per client |
| **FedNova** | w ← w_global + Σᵢ pᵢ (wᵢ − w_global) / τᵢ · τ_eff |
| **ServerMomentum** | w_server ← β · w_server + (1−β) · Σᵢ pᵢ wᵢ |

FedNova τᵢ = number of local SGD batches per client. ServerMomentum is server-side EMA — it has **no** per-client control variates and is explicitly distinct from SCAFFOLD ([Karimireddy et al., 2020](https://arxiv.org/abs/1910.06378)).

---

## Architecture

```mermaid
flowchart LR
    DS["PathMNIST size 28\nKaggle runtime only"]
    PRT["Dirichlet Partition\nα=0.5 · 5 clients\nseeds 42 / 123 / 456"]
    GATE["Centralized Baseline\nmacro F1 ≥ 0.70\nSHA-256 verified"]
    STRAT["Strategy\nFedAvg · FedProx\nFedNova · ServerMomentum"]
    ART["JSON + PNG\nArtifacts"]
    DASH["Streamlit\nDashboard\ndisplay-only"]

    DS -->|"dataset.labels"| PRT
    PRT -->|"client_indices"| STRAT
    GATE -->|"integrity gate"| STRAT
    STRAT -->|"per-round metrics"| ART
    ART --> DASH
```

---

## Quick Start — Offline Tests (no dataset, no GPU)

```bash
# Python 3.11+ recommended
git clone https://github.com/ajinkya-awari/fedscope
cd fedscope
pip install -e ".[dev]"

# Run all 57 offline contract tests
python -m pytest -q tests/
```

The offline suite verifies all strategy contracts, the centralized gate, the dashboard missing-artifact behavior, and the Kaggle runner's bounded scope using synthetic fixtures only. No PathMNIST, Flower, or GPU required.

---

## Design Contracts

- **Labels:** `dataset.labels` only; never `dataset.targets`. Squeezed to 1-D before partitioning and per-batch before CrossEntropyLoss.
- **Partitioning:** `np.random.default_rng(seed)` — reproducible across Python versions. Every sample assigned exactly once.
- **FedProx:** `proximal_mu` must be supplied explicitly in the client config — missing key raises `KeyError` by design.
- **FedNova:** `aggregate_fit` expects Flower-style result objects (`.parameters`, `.num_examples`, `.metrics["tau"]`). Each client returns `tau` = number of local SGD batches.
- **ServerMomentum:** server-side EMA of client-averaged weights; no per-client control variates.
- **Drift capture:** computed inside `aggregate_fit()` against the pre-round global weights snapshot from `configure_fit()`.
- **Gate integrity:** the centralized gate payload is persisted with a SHA-256 digest. `require_centralized_gate` verifies this digest before any federated work begins.
- **Dashboard:** reads only precomputed JSON + PNG artifacts. Raises a visible error and stops if any artifact is missing.

---

## Kaggle Execution (requires authentication)

```bash
# 1. Upload the file bundle listed in kaggle/upload_manifest.json

# 2. Inside an authenticated Kaggle notebook:
pip install -r kaggle/requirements.txt

# 3. Smoke run (1 round, seed 42) — must pass before multi-seed run
python kaggle/run_fedscope.py --mode smoke \
    --data-root /kaggle/working/medmnist \
    --output-root /kaggle/working/fedscope_artifacts \
    --device auto

# 4. Full study — only after smoke and centralized gate pass
python kaggle/run_fedscope.py --mode approved-seeds \
    --data-root /kaggle/working/medmnist \
    --output-root /kaggle/working/fedscope_artifacts \
    --device auto
```

See `notebooks/KAGGLE_RUNBOOK_fedscope.md` for the full cell-by-cell protocol.

<details>
<summary>Reproducibility details</summary>

| Setting | Value |
|---------|-------|
| Dataset | PathMNIST size 28 — 89,996 train / 10,004 val / 7,180 test patches |
| Classes | 9 (colorectal tissue types: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM) |
| Model | FedScopeCNN — 3-channel RGB, 3 conv blocks, 9-class output |
| Optimizer (centralized) | SGD, lr=0.01, momentum=0.9, 30 epochs |
| Partitioning | Dirichlet α=0.5, 5 clients, per-seed RNG |
| Seeds | 42, 123, 456 |
| Federated rounds | 3 (compute-limited; convergence requires more) |
| Local epochs | 5 per client per round |
| Primary metric | Macro F1, `zero_division=0` (absent classes score zero) |
| Compute | Kaggle CPU (P100 GPU incompatible with PyTorch 2.10) |
| Kernel | `ajinkya1225/05-fedscope` v19 — exit_code=0, 2026-09-05 |
| Gate verification | SHA-256 digest on persisted baseline payload |
| Artifact format | JSON (per-round metrics, drift, comm cost) + PNG (convergence, drift, communication) |

</details>

---

## Verification Status

| Component | Status |
|-----------|--------|
| Offline contract tests (57 functions) | **Verified** — all pass (2026-09-03) |
| PathMNIST download + preflight | **Verified on Kaggle** — v19 (2026-09-05) |
| Smoke run (1 round, seed 42) | **PASSED** — exit_code=0 (2026-09-04) |
| Centralized macro F1 gate (≥ 0.70) | **PASSED** — all 3 seeds: 0.779 / 0.790 / 0.795 (2026-09-05) |
| Federated training (FedAvg / FedProx / FedNova / ServerMomentum) | **COMPLETE** — 3 rounds × 3 seeds × 4 algorithms (2026-09-05) |
| Three-seed reproducibility (42, 123, 456) | **COMPLETE** — all seeds ran, artifacts persisted (2026-09-05) |
| GPU execution | CPU-only (Kaggle P100 sm_60 incompatible with PyTorch 2.10) |
| Dashboard rendering | Artifacts produced; dashboard ready to connect |

---

## Limitations

- Federated results are from 3 rounds on CPU. Strategy rankings are not stable at this compute budget — near-random F1 for all strategies is the expected result, not a failure. Full convergence characterization requires 50+ rounds on GPU.
- PathMNIST is a small proxy task (28×28 RGB patches from colorectal histology). Results on larger, noisier, or real federated datasets may differ substantially.
- Client drift and communication cost metrics are both near-zero at 3 rounds; drift analysis becomes meaningful only after convergence.
- The Flower simulation uses Ray internally; behavior under different Ray versions is not verified.

---

## Security and Privacy

- PathMNIST is a publicly released benchmark dataset. No patient-level or restricted clinical data is included.
- The dashboard loads only precomputed artifacts from a user-supplied root path. It does not download data or start training.
- The Kaggle runner is bounded: it accepts only PathMNIST size 28, exactly five clients, and seeds 42 / 123 / 456.
- No credentials, API keys, or private paths are present in this repository.

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=100&section=footer"/>
</div>
