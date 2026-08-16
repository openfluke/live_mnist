# live_mnist

Live mid-stream adaptation benchmark on **MNIST 80/20**. One host of the
[`tide`](../tide) serve+train engine (any dataset that implements `runner.Dataset`
works the same way). Measuring aligned with
[`test41_w_sine_ada_perm`](../loom/arcagitesting/test41_w_sine_ada_perm).

---

## What you are measuring

Not “best static MNIST accuracy.” Which **dtype × quant × train mode × arch**
still **serves answers while learning** under mid-stream label flips — and how
expensive that is in storage.

| Axis | Metrics | Meaning |
|------|---------|---------|
| Adaptation quality | SoftAcc, AdaptPct, Stability, Consistency | Track quality / recovery after phase A→B→A2 |
| Duty-cycle availability | Availability, ZeroDowntime | InferMs / (InferMs+TrainMs) |
| Cost | WeightBytes, HeapBytes, MobileScore | Model size + Score/MiB |

### Pareto front

Improving Score often costs RAM; boosting SoftAcc often costs Availability.
The undominated edge of that tradeoff surface is the result — not a single
winner column.

---

## Lucy / test41 formulas

```
SoftAcc      = SoftAcc on true-class softmax prob vs 1.0 (scale 1 → ≈100×p)
             // sine test41 uses scale 0.10 on continuous [0,1] targets; raw logits vs 1.0
             // wrongly zeroed SoftAcc (and Score) while Hard Acc was already high
Availability = InferMs / (InferMs + TrainMs) × 100
Score        = Throughput × Availability × SoftAcc / 10_000
MobileScore  = Score / WeightMiB
```

Hard argmax Acc is still recorded (`avg_accuracy`) but **Score uses SoftAcc**.

Mid-stream flip: phase **A** → **B** (`label = (label+5)%10`) → **A2**.

---

## Network

**cnn** (single×1)
```
input [B,1,28,28]
  → CNN2 (8 filters, k=3, s=1, p=1, ReLU)
  → CNN2 (16 filters, k=3, s=2, p=1, ReLU)
  → flatten
  → Dense → 10 logits
```

**bicameral** (×2) / **tricameral** (×3)
```
…same CNN stack…
  → Dense (flat → 64)
  → Parallel(n×Dense, add)
  → Dense → 10 logits
```

Backend: **SIMD only**. Train split defaults to a **class-balanced 8000** of the 80% official train (pass `-train-n 0` for all ~48000). Val stays the full 20% for serve.

---

## Run

```bash
go run . -addr 0.0.0.0:8080
# default: -mode full, -train-n 8000
```

A **full** matrix is ~3700 cells. At a full 48k epoch that is on the order of **a month** on this box (~10 min/cell). Cut wall time with:

| Lever | Flag | Effect |
|-------|------|--------|
| Shorter epoch | `-train-n 8000` (default) | ~6× faster per cell; A→B→A2 flips still at 1/3 and 2/3 |
| CNN-only screen | `-mode screen` | Lucy 6 × cnn × all dtypes/quants (~324 cells) |
| One arch | `-arches cnn` | skip bi/tri on whatever `-mode` you picked |
| Tiny probe | `-mode smoke` | few dtypes × all modes × all arches |

Resume still skips finished cell IDs. If an inflight cell is past the new `-train-n`, pass `-train-n 0` to finish it (or `-fresh`).

| Flag | Default | Meaning |
|------|---------|---------|
| `-mode` | `full` | `full` \| `screen` \| `smoke` \| `kquant` |
| `-train-n` | `8000` | class-balanced train examples per cell (`0` = all ~48000) |
| `-arches` | all in mode | `cnn`, `bicameral`, `tricameral` (comma list) |
| `-batch` | `4` | permutations per dashboard batch |
| `-micro` | `8` | MNIST micro-batch |
| `-data` | `data` | MNIST download cache |
| `-ckpt` | `checkpoint` | progress + model weights |
| `-ckpt-sec` | `60` | save scores + inflight model every N seconds |
| `-fresh` | off | ignore checkpoint and start clean at epoch 1 |

**Default:** each permutation trains **one pass** over `-train-n` examples (class-balanced).  
Re-run after the sweep finishes → **epoch N+1**. Ctrl+C mid-epoch resumes.

### Train modes (Lucy 6 + full Welvet named set @ SIMD)

Checkpoint-stable Lucy tokens: `sgd`, `step_sgd`, `tween`, `tween_chain`, `step_tween`, `step_tween_chain`.  
Then every other `parallel.AllNamedTrainModes()` name (Split, Alt, HeadProxy, FastProxy, Linear, Sparse, Mesh*, …).

× arches `cnn` (single×1) \| `bicameral` (×2) \| `tricameral` (×3).

**Resume:** re-run `go run .` and press **Resume** on the dashboard. Finished cell IDs are skipped — new modes/arches only train the new IDs. Do not pass `-fresh` unless you want to wipe epoch 1.

**Removed:** `tween_head`, CPU-tiled twin modes, `*_simd` suffixes (everything is SIMD).

`full` = all dtypes × all packs × all modes × all arches.  
`screen` = Lucy 6 × cnn × all dtypes/packs (first pass).  
`smoke` = few dtypes/packs × all modes × all arches.  
`kquant` = Q2_K…Q6_K × Lucy modes × arches.

---

## Stack

- **tide** — serve+train runner, Lucy/test41 metrics, live HTML pulse  
- **Welvet** — CNN2 / Dense / Parallel, FormatNone dtypes, k-quants, SIMD  
- Measuring lineage: test41-w perm SoftAcc + duty-cycle Availability  
