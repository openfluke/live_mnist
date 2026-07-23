// live_mnist — 80/20 MNIST adaptation sweep via tide + Welvet.
//
// Default: each permutation trains one full epoch over the 80% train split.
// Re-run after a finished sweep starts epoch N+1 (weights continue).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/openfluke/live_mnist/mnist"
	"github.com/openfluke/tide/checkpoint"
	"github.com/openfluke/tide/dash"
	"github.com/openfluke/tide/permute"
	"github.com/openfluke/tide/pulse"
	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/simd"
)

func main() {
	addr := flag.String("addr", ":8080", "dashboard listen address")
	mode := flag.String("mode", "full", "permutation set: full | smoke | kquant (default: full matrix)")
	dataDir := flag.String("data", "data", "MNIST cache directory")
	ckptDir := flag.String("ckpt", "checkpoint", "progress + model checkpoint directory")
	batch := flag.Int("batch", 4, "permutations per dashboard batch")
	micro := flag.Int("micro", 8, "MNIST micro-batch size")
	ckptSec := flag.Int("ckpt-sec", 60, "seconds between model/score checkpoints")
	lr := flag.Float64("lr", 0.02, "learning rate")
	fresh := flag.Bool("fresh", false, "ignore existing checkpoint and start clean at epoch 1")
	flag.Parse()

	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println(" live_mnist — tide × Welvet mid-stream adaptation")
	fmt.Println(" input → cnn2 → cnn2 → dense → 10")
	fmt.Println(" Score = Throughput × Availability% × AvgAccuracy% / 10_000")
	fmt.Println(" Default: full dtype × quant × train-mode matrix")
	fmt.Println("          1 epoch over 80% train per permutation")
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Printf(" SIMD linked: %v\n", simd.Enabled())
	fmt.Printf(" Dashboard:   http://127.0.0.1%s\n", *addr)
	fmt.Printf(" Mode:        %s\n", *mode)
	fmt.Printf(" Checkpoint:  %s (every %ds)\n\n", *ckptDir, *ckptSec)

	fmt.Println("Loading MNIST (80/20 train/val from official train set)…")
	split, err := mnist.Load(*dataDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("  train=%d  val=%d  test=%d\n", len(split.Train), len(split.Val), len(split.Test))
	fmt.Printf("  → each cell trains until all %d train examples are seen once\n\n", len(split.Train))

	var pcfg permute.Config
	switch *mode {
	case "full":
		pcfg = permute.Full()
	case "kquant":
		pcfg = permute.KQuant()
	default:
		pcfg = permute.Smoke()
	}
	cells := permute.Expand(pcfg)
	fmt.Printf(" Permutations: %d (batch size %d)\n", len(cells), *batch)

	store := checkpoint.New(*ckptDir, *mode)
	var resume *checkpoint.Progress
	if !*fresh {
		resume, err = store.Load()
		if err != nil {
			fmt.Fprintln(os.Stderr, "checkpoint load:", err)
			os.Exit(1)
		}
	}
	epoch, resume := checkpoint.PrepareEpoch(resume, cells)
	if resume != nil {
		done := checkpoint.DoneSet(resume)
		fmt.Printf("  epoch %d — resumed: %d/%d cells done", epoch, len(done), len(cells))
		if resume.Inflight != nil {
			fmt.Printf(", inflight=%s @%d/%d", resume.Inflight.Cell.ID, resume.Inflight.TrainOffset, len(split.Train))
		}
		fmt.Println()
		printBests("  checkpoint bests", resume.Best)
	} else {
		fmt.Printf("  epoch %d — fresh sweep\n", epoch)
	}

	tr := pulse.New()
	srv := &dash.Server{Tracker: tr, Addr: *addr}
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			fmt.Fprintln(os.Stderr, "dash:", err)
		}
	}()

	cfg := runner.DefaultConfig(cells)
	cfg.BatchSize = *batch
	cfg.Epoch = epoch
	cfg.CheckpointEvery = time.Duration(*ckptSec) * time.Second
	cfg.LR = *lr
	cfg.Store = store
	cfg.Resume = resume

	ds := newTideDS(split, *micro, 0x4D4E4953)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Printf("Running epoch %d — open the dashboard to watch pulses.\n", epoch)
	if err := runner.Run(ctx, cfg, ds, tr); err != nil && ctx.Err() == nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	live := tr.Snapshot()
	printBests("\n── Best (3 metrics + Score) ──", live.Best)
	fmt.Println("\n── Leaderboard (top by Lucy Score) ──")
	for i, r := range live.Leaderboard {
		if i >= 15 {
			break
		}
		s := r.Snapshot
		fmt.Printf("%2d  [e%d] %-42s  score=%7.3f  acc=%5.1f  thru=%7.1f  avail=%5.1f  %s\n",
			i+1, r.Epoch, r.Cell.ID, s.Score, s.AvgAccuracy, s.Throughput, s.Availability, r.Status)
	}
	if ctx.Err() != nil {
		fmt.Printf("\nStopped — progress saved under %s (re-run to resume mid-epoch).\n", *ckptDir)
		return
	}
	fmt.Printf("\nEpoch %d complete. Re-run `go run .` for epoch %d (weights continue).\n", epoch, epoch+1)
	fmt.Println("Dashboard still serving — Ctrl+C to exit.")
	<-ctx.Done()
}

func printBests(title string, b pulse.Best) {
	fmt.Println(title)
	printBestLine("score", b.Score)
	printBestLine("throughput", b.Throughput)
	printBestLine("availability", b.Availability)
	printBestLine("accuracy", b.Accuracy)
}

func printBestLine(name string, r *pulse.Result) {
	if r == nil {
		fmt.Printf("  %-12s  —\n", name)
		return
	}
	s := r.Snapshot
	fmt.Printf("  %-12s  %s  score=%.3f acc=%.1f thru=%.1f avail=%.1f\n",
		name, r.Cell.ID, s.Score, s.AvgAccuracy, s.Throughput, s.Availability)
}
