// live_mnist — 80/20 MNIST adaptation sweep via tide + Welvet.
//
// Architecture: input → CNN2 → CNN2 → Dense → 10
// Score: Throughput × Availability% × AvgAccuracy% / 10_000  (Lucy dense equation)
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
	mode := flag.String("mode", "smoke", "permutation set: smoke | kquant | full")
	dataDir := flag.String("data", "data", "MNIST cache directory")
	ckptDir := flag.String("ckpt", "checkpoint", "progress + model checkpoint directory")
	batch := flag.Int("batch", 4, "permutations per dashboard batch")
	micro := flag.Int("micro", 8, "MNIST micro-batch size")
	cellSec := flag.Int("cell-sec", 12, "seconds per permutation cell")
	ckptSec := flag.Int("ckpt-sec", 60, "seconds between model/score checkpoints")
	lr := flag.Float64("lr", 0.02, "learning rate")
	fresh := flag.Bool("fresh", false, "ignore existing checkpoint and start clean")
	flag.Parse()

	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println(" live_mnist — tide × Welvet mid-stream adaptation")
	fmt.Println(" input → cnn2 → cnn2 → dense → 10")
	fmt.Println(" Score = Throughput × Availability% × AvgAccuracy% / 10_000")
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
	fmt.Printf("  train=%d  val=%d  test=%d\n\n", len(split.Train), len(split.Val), len(split.Test))

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
		if resume != nil {
			if resume.Mode != "" && resume.Mode != *mode {
				fmt.Printf("  warning: checkpoint mode=%q != -mode %q (still resuming)\n", resume.Mode, *mode)
			}
			fmt.Printf("  resumed: %d completed, next=%d", len(resume.Completed), resume.NextCellIndex)
			if resume.Inflight != nil {
				fmt.Printf(", inflight=%s (%.1fs elapsed)", resume.Inflight.Cell.ID, float64(resume.Inflight.ElapsedNS)/1e9)
			}
			fmt.Println()
			printBests("  checkpoint bests", resume.Best)
		}
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
	cfg.CellDuration = time.Duration(*cellSec) * time.Second
	cfg.CheckpointEvery = time.Duration(*ckptSec) * time.Second
	cfg.LR = *lr
	cfg.Store = store
	cfg.Resume = resume

	ds := newTideDS(split, *micro, 0x4D4E4953)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Println("Running adaptation sweep — open the dashboard to watch pulses.")
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
		fmt.Printf("%2d  %-48s  score=%7.3f  acc=%5.1f  thru=%7.1f  avail=%5.1f  %s\n",
			i+1, r.Cell.ID, s.Score, s.AvgAccuracy, s.Throughput, s.Availability, r.Status)
	}
	if ctx.Err() != nil {
		fmt.Printf("\nStopped — progress saved under %s (re-run to resume).\n", *ckptDir)
		return
	}
	fmt.Println("\nDone. Dashboard still serving — Ctrl+C to exit.")
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
