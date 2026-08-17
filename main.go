// live_mnist — 80/20 MNIST adaptation sweep via tide + Welvet.
//
// Default: each permutation trains a class-balanced 8000-example pass
// (-train-n 0 for the full 80% split). Re-run after a finished sweep starts
// epoch N+1 (weights continue).
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
	mode := flag.String("mode", "full", "permutation set: full | screen | smoke | kquant")
	arches := flag.String("arches", "", "limit arches: comma list single,bicameral,tricameral (cnn still accepted)")
	trainN := flag.Int("train-n", 8000, "train examples per cell, class-balanced (0 = all ~48000)")
	dataDir := flag.String("data", "data", "MNIST cache directory")
	ckptDir := flag.String("ckpt", "checkpoint", "progress + model checkpoint directory")
	batch := flag.Int("batch", 4, "permutations per dashboard batch")
	micro := flag.Int("micro", 8, "MNIST micro-batch size")
	ckptSec := flag.Int("ckpt-sec", 60, "seconds between model/score checkpoints")
	lr := flag.Float64("lr", 0.02, "learning rate")
	fresh := flag.Bool("fresh", false, "ignore existing checkpoint and start clean at epoch 1")
	autostart := flag.Bool("autostart", false, "start training immediately (skip dashboard Start button)")
	flag.Parse()

	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println(" live_mnist — tide × Welvet mid-stream adaptation @ SIMD")
	fmt.Println(" arch: single×1 | bicameral×2 | tricameral×3")
	fmt.Println(" modes: Lucy 6 (sgd/step/tween…) + Welvet credit (Split/FastProxy/Sparse/Mesh*…)")
	fmt.Println(" Score = Throughput × Availability × SoftAcc / 10_000")
	fmt.Println(" Availability = InferMs / (InferMs + TrainMs)")
	fmt.Println(" Default train-n=8000 class-balanced (not a full 48k epoch)")
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
	fullTrain := len(split.Train)
	split.Train = mnist.TakeBalanced(split.Train, *trainN, 0x4D4E4953)
	fmt.Printf("  train=%d", len(split.Train))
	if len(split.Train) < fullTrain {
		fmt.Printf("/%d class-balanced (-train-n %d; 0 = all)", fullTrain, *trainN)
	}
	fmt.Printf("  val=%d  test=%d\n", len(split.Val), len(split.Test))
	fmt.Printf("  → each cell trains until all %d train examples are seen once (flips at 1/3 and 2/3)\n\n", len(split.Train))

	var pcfg permute.Config
	switch *mode {
	case "full":
		pcfg = permute.Full()
	case "screen":
		pcfg = permute.Screen()
	case "kquant":
		pcfg = permute.KQuant()
	case "smoke":
		pcfg = permute.Smoke()
	default:
		fmt.Fprintf(os.Stderr, "unknown -mode %q (full|screen|smoke|kquant)\n", *mode)
		os.Exit(2)
	}
	if list := parseArches(*arches); len(list) > 0 {
		pcfg.Arches = list
	}
	cells := permute.Expand(pcfg)
	fmt.Printf(" Permutations: %d (batch size %d)\n", len(cells), *batch)
	fmt.Printf(" Rough ETA:    %s  (from ~10 min/cell at 48k on this box, scaled by train-n)\n",
		roughETA(len(cells), len(split.Train), fullTrain))
	if *mode == "full" && (len(pcfg.Arches) != 1) {
		fmt.Println(" Tip:          -mode screen  (Lucy 6 × single × all dtypes) or -arches single to cut the month down")
	}

	store := checkpoint.New(*ckptDir, *mode)
	var resume *checkpoint.Progress
	if !*fresh {
		resume, err = store.Load()
		if err != nil {
			fmt.Fprintln(os.Stderr, "checkpoint load:", err)
			os.Exit(1)
		}
	}
	if resume != nil && resume.Inflight != nil && resume.Inflight.TrainOffset > len(split.Train) {
		fmt.Fprintf(os.Stderr, "inflight cell %s is at example %d but this run only has %d train examples.\n",
			resume.Inflight.Cell.ID, resume.Inflight.TrainOffset, len(split.Train))
		fmt.Fprintf(os.Stderr, "Pass -train-n 0 to finish it, or -fresh to drop inflight.\n")
		os.Exit(2)
	}
	epoch, resume := checkpoint.PrepareEpoch(resume, cells)
	if resume != nil {
		done := checkpoint.DoneSet(resume)
		fmt.Printf("  epoch %d — resumed: %d/%d cells done", epoch, len(done), len(cells))
		if resume.Inflight != nil {
			fmt.Printf(", inflight=%s @%d/%d", resume.Inflight.Cell.ID, resume.Inflight.TrainOffset, len(split.Train))
		}
		fmt.Println()
		if len(split.Train) < fullTrain && len(done) > 0 {
			fmt.Println("  note: finished cells may have used a longer epoch; new cells use -train-n. Scores mix two lengths.")
		}
		printBests("  checkpoint bests (raw)", resume.Best)
		printMobile("  checkpoint bests (mobile = metric/MiB)", resume.BestMobile)
	} else {
		fmt.Printf("  epoch %d — fresh sweep\n", epoch)
	}

	tr := pulse.New()
	srv := &dash.Server{
		Tracker:  tr,
		Cells:    cells,
		Addr:     *addr,
		Epoch:    epoch,
		Task:     "MNIST",
		Subtitle: fmt.Sprintf("%d train examples · mid-stream flips A→B→A2 · SIMD", len(split.Train)),
	}
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

	// Load boards first so the dash shows metrics while paused.
	doneN := len(checkpoint.DoneSet(resume))
	runner.Hydrate(tr, cfg, fmt.Sprintf(
		"paused — epoch %d — %d/%d done — press Start on dashboard",
		epoch, doneN, len(cells)))

	if *autostart {
		srv.SignalStart()
		fmt.Printf("Autostart — running epoch %d.\n", epoch)
	} else {
		fmt.Printf("Dashboard ready (epoch %d) — open it and press Start/Resume.\n", epoch)
		fmt.Printf("  finished cells are skipped; new Welvet modes/arches only run the new IDs\n")
		fmt.Printf("  (or re-run with -autostart)\n")
		if err := srv.AwaitStart(ctx); err != nil {
			fmt.Printf("\nStopped before start — checkpoint unchanged under %s.\n", *ckptDir)
			return
		}
		fmt.Printf("Start pressed — running epoch %d.\n", epoch)
	}

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
		fmt.Printf("%2d  [e%d] %-48s  score=%7.3f  soft=%5.1f  hard=%5.1f  thru=%7.1f  avail=%5.1f  adapt=%5.1f  %6.1fKiB  %s\n",
			i+1, r.Epoch, r.Cell.ID, s.Score, s.SoftAcc, s.AvgAccuracy, s.Throughput, s.Availability,
			s.AdaptPct, float64(s.WeightBytes)/1024, r.Status)
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

func parseArches(s string) []permute.ArchKind {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var out []permute.ArchKind
	for _, p := range strings.Split(s, ",") {
		p = strings.ToLower(strings.TrimSpace(p))
		switch p {
		case "":
			continue
		case "cnn", "single":
			out = append(out, permute.ArchSingle)
		case "bicameral", "bi":
			out = append(out, permute.ArchBicameral)
		case "tricameral", "tri":
			out = append(out, permute.ArchTricameral)
		default:
			fmt.Fprintf(os.Stderr, "unknown arch %q (single|bicameral|tricameral)\n", p)
			os.Exit(2)
		}
	}
	return out
}

// roughETA scales ~10 min/cell at the full 80% split (measured on this machine's checkpoint).
func roughETA(cells, trainN, fullN int) string {
	if cells < 1 {
		return "—"
	}
	if fullN < 1 {
		fullN = 48000
	}
	if trainN < 1 {
		trainN = fullN
	}
	sec := float64(cells) * 600 * float64(trainN) / float64(fullN)
	switch {
	case sec < 90:
		return fmt.Sprintf("~%.0fs", sec)
	case sec < 2*3600:
		return fmt.Sprintf("~%.0f min", sec/60)
	default:
		return fmt.Sprintf("~%.1f days", sec/86400)
	}
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
	fmt.Printf("  %-12s  %s  score=%.3f soft=%.1f hard=%.1f thru=%.1f avail=%.1f adapt=%.1f  (%.1f KiB)\n",
		name, r.Cell.ID, s.Score, s.SoftAcc, s.AvgAccuracy, s.Throughput, s.Availability, s.AdaptPct, float64(s.WeightBytes)/1024)
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
