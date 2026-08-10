// live_mnist — 80/20 MNIST adaptation sweep via tide + Welvet.
//
// Default: each permutation trains one full epoch over the 80% train split.
// Re-run after a finished sweep starts epoch N+1 (weights continue).
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"strings"
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
	addr := flag.String("addr", "0.0.0.0:8080", "dashboard listen address (0.0.0.0 = all interfaces)")
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
	fmt.Printf(" Dashboard:   %s\n", dashURLs(*addr))
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
		printBests("  checkpoint bests (raw)", resume.Best)
		printMobile("  checkpoint bests (mobile = metric/MiB)", resume.BestMobile)
	} else {
		fmt.Printf("  epoch %d — fresh sweep\n", epoch)
	}

	tr := pulse.New()
	srv := &dash.Server{Tracker: tr, Cells: cells, Addr: *addr}
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
	printBests("\n── Best raw (3 metrics + Score) ──", live.Best)
	printMobile("\n── Best mobile (metric / MiB) ──", live.BestMobile)
	fmt.Println("\n── Leaderboard raw (Lucy Score) ──")
	for i, r := range live.Leaderboard {
		if i >= 15 {
			break
		}
		s := r.Snapshot
		fmt.Printf("%2d  [e%d] %-42s  score=%7.3f  acc=%5.1f  thru=%7.1f  avail=%5.1f  %6.1fKiB  %s\n",
			i+1, r.Epoch, r.Cell.ID, s.Score, s.AvgAccuracy, s.Throughput, s.Availability,
			float64(s.WeightBytes)/1024, r.Status)
	}
	fmt.Println("\n── Leaderboard mobile (Score / MiB) ──")
	for i, r := range live.LeaderboardMobile {
		if i >= 15 {
			break
		}
		s := r.Snapshot
		fmt.Printf("%2d  [e%d] %-42s  score/MiB=%8.2f  score=%7.3f  acc=%5.1f  %6.1fKiB  %s\n",
			i+1, r.Epoch, r.Cell.ID, s.MobileScore, s.Score, s.AvgAccuracy,
			float64(s.WeightBytes)/1024, r.Status)
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
	fmt.Printf("  %-12s  %s  score=%.3f acc=%.1f thru=%.1f avail=%.1f  (%.1f KiB)\n",
		name, r.Cell.ID, s.Score, s.AvgAccuracy, s.Throughput, s.Availability, float64(s.WeightBytes)/1024)
}

func printMobile(title string, b pulse.BestMobile) {
	fmt.Println(title)
	printMobileLine("score", b.Score, func(s pulse.Result) float64 { return s.Snapshot.MobileScore })
	printMobileLine("throughput", b.Throughput, func(s pulse.Result) float64 { return s.Snapshot.MobileThroughput })
	printMobileLine("availability", b.Availability, func(s pulse.Result) float64 { return s.Snapshot.MobileAvailability })
	printMobileLine("accuracy", b.Accuracy, func(s pulse.Result) float64 { return s.Snapshot.MobileAccuracy })
}

func printMobileLine(name string, r *pulse.Result, eff func(pulse.Result) float64) {
	if r == nil {
		fmt.Printf("  %-12s  —\n", name)
		return
	}
	s := r.Snapshot
	fmt.Printf("  %-12s  %s  eff=%.3f/MiB  raw score=%.3f acc=%.1f thru=%.1f  (%.1f KiB)\n",
		name, r.Cell.ID, eff(*r), s.Score, s.AvgAccuracy, s.Throughput, float64(s.WeightBytes)/1024)
}

// dashURLs prints local + LAN URLs when bound on all interfaces.
func dashURLs(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		// bare ":8080"
		if strings.HasPrefix(addr, ":") {
			host, port = "", strings.TrimPrefix(addr, ":")
		} else {
			return "http://" + addr
		}
	}
	if port == "" {
		port = "8080"
	}
	all := host == "" || host == "0.0.0.0" || host == "::"
	if !all {
		return "http://" + net.JoinHostPort(host, port)
	}
	lines := []string{
		fmt.Sprintf("http://127.0.0.1:%s  (local)", port),
		fmt.Sprintf("listening on 0.0.0.0:%s — remote: http://<this-host-ip>:%s", port, port),
	}
	if ip := firstLANIPv4(); ip != "" {
		lines = append(lines, fmt.Sprintf("http://%s:%s  (LAN)", ip, port))
	}
	return strings.Join(lines, "\n              ")
}

func firstLANIPv4() string {
	ifaces, err := net.Interfaces()
	if err != nil {
		return ""
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			var ip net.IP
			switch v := a.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip == nil || ip.IsLoopback() {
				continue
			}
			ip = ip.To4()
			if ip == nil {
				continue
			}
			return ip.String()
		}
	}
	return ""
}
