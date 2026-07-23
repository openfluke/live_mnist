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
go run . -addr :8080 -mode smoke
# open http://127.0.0.1:8080
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-mode` | `smoke` | `smoke` \| `kquant` \| `full` |
| `-cell-sec` | `12` | wall time per dtype×format×mode cell |
| `-batch` | `4` | permutations per dashboard batch |
| `-micro` | `8` | MNIST micro-batch |
| `-data` | `data` | MNIST download cache |
| `-ckpt` | `checkpoint` | progress + model weights |
| `-ckpt-sec` | `60` | save scores + inflight model every N seconds |
| `-fresh` | off | ignore checkpoint and start clean |

### Checkpoint / resume

Every minute (configurable) tide writes:

- `checkpoint/progress.json` — completed cells, Lucy scores, **best score / throughput / availability / accuracy**, inflight cell state  
- `checkpoint/models/inflight/` — current cell weights (cnn1/cnn2/head `.bin`)  
- `checkpoint/models/best_{score,throughput,availability,accuracy}/` — best models per axis  
- `checkpoint/models/<cell_id>/` — finished cell weights  

Stop (Ctrl+C) and re-run the same command to resume from the next unfinished cell (or mid-cell inflight). Use `-fresh` to wipe resume state (delete `checkpoint/` manually if you want a clean slate on disk).

`smoke` covers a few dtypes + classic/k-quants × SGD/tween_head × SIMD.  
`kquant` sweeps Q2_K…Q6_K.  
`full` expands Welvet's full dtype × format matrix (long).

## Stack

- **tide** — serve+train runner, Lucy metrics, live HTML pulse  
- **Welvet** — native CNN2 / Dense, FormatNone dtypes, k-quants, SIMD backends (same training path as down-the-dem)
