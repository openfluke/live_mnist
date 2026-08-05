# live_mnist

Live mid-stream adaptation benchmark on **MNIST 80/20**, driven by [`tide`](../tide) on Welvet.

## Network

```
input [B,1,28,28]
  → CNN2 (8 filters, k=3, s=1, p=1, ReLU)
  → CNN2 (16 filters, k=3, s=2, p=1, ReLU)
  → flatten
  → Dense → 10 logits
```

Train split = 80% of official MNIST train; val = remaining 20% (every 5th index). Serve draws from val; train from the 80% pool.

## Score (Lucy dense equation)

```
Score = Throughput × Availability% × AvgAccuracy% / 10_000
```

Mid-stream flip: phase **A** (normal labels) → **B** (`label = (label+5)%10`) → **A2** (normal again).

## Run

```bash
go run . -addr 0.0.0.0:8080
# local:  http://127.0.0.1:8080
# remote: http://<host-lan-ip>:8080
# default -mode full = all dtypes × all packs/k-quants × all train modes
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-mode` | `full` | `full` \| `smoke` \| `kquant` |
| `-batch` | `4` | permutations per dashboard batch |
| `-micro` | `8` | MNIST micro-batch |
| `-data` | `data` | MNIST download cache |
| `-ckpt` | `checkpoint` | progress + model weights |
| `-ckpt-sec` | `60` | save scores + inflight model every N seconds |
| `-fresh` | off | ignore checkpoint and start clean at epoch 1 |

**Default:** each permutation trains **one full epoch** over the 80% train split (all 48k examples once).  
Re-run after the sweep finishes → **epoch N+1** (loads prior cell weights). Ctrl+C mid-epoch resumes at the saved train offset.

### Checkpoint / resume

Every minute (configurable) tide writes:

- `checkpoint/progress.json` — completed cells, Lucy scores, **best score / throughput / availability / accuracy**, inflight cell state  
- `checkpoint/history.json` — full pulse timeline for the dashboard (refresh / other machines)  
- `checkpoint/models/inflight/` — current cell weights (cnn1/cnn2/head `.bin`)  
- `checkpoint/models/best_{score,throughput,availability,accuracy}/` — best raw models per axis  
- `checkpoint/models/best_mobile_{score,throughput,availability,accuracy}/` — best **metric/MiB** (mobile)  
- `checkpoint/models/<cell_id>/` — finished cell weights  

Stop (Ctrl+C) and re-run the same command to resume from the next unfinished cell (or mid-cell inflight). Use `-fresh` to wipe resume state (delete `checkpoint/` manually if you want a clean slate on disk).

Dashboard loads **server history** on every poll — refresh or open from another machine and you still see the chart + can scrub back in time.

### Train modes (Lucy suite)

`sgd`, `step_sgd`, `tween`, `tween_chain`, `step_tween`, `step_tween_chain` (+ `_simd` each), plus `tween_head` baselines — crossed with dtypes/quants.

`full` (default) = all 34 dtypes × all Welvet packs/k-quants × all train modes.  
`smoke` = few dtypes/packs × all modes (quick check).  
`kquant` = Q2_K…Q6_K × Lucy modes only.

## Stack

- **tide** — serve+train runner, Lucy metrics, live HTML pulse  
- **Welvet** — native CNN2 / Dense, FormatNone dtypes, k-quants, SIMD backends (same training path as down-the-dem)
