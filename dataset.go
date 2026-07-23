package main

import (
	"math/rand/v2"

	"github.com/openfluke/live_mnist/mnist"
	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/core"
)

// tideDS wraps the 80/20 MNIST split for tide's serve+train loops.
// Serve draws from Val; train draws from Train.
type tideDS struct {
	train []mnist.Image
	val   []mnist.Image
	batch int
	rng   *rand.Rand
}

func newTideDS(split *mnist.Split, batch int, seed uint64) *tideDS {
	if batch < 1 {
		batch = 8
	}
	return &tideDS{
		train: split.Train,
		val:   split.Val,
		batch: batch,
		rng:   rand.New(rand.NewPCG(seed, seed^0xC0FFEE)),
	}
}

func (d *tideDS) NextServe(phase string) runner.Sample {
	_ = phase
	return d.batchFrom(d.val)
}

func (d *tideDS) NextTrain(phase string) runner.Sample {
	_ = phase
	return d.batchFrom(d.train)
}

func (d *tideDS) batchFrom(pool []mnist.Image) runner.Sample {
	n := d.batch
	if n > len(pool) {
		n = len(pool)
	}
	x := core.NewTensor[float32](n, 1, 28, 28)
	target := core.NewTensor[float32](n, 10)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		im := pool[d.rng.IntN(len(pool))]
		labels[i] = im.Label
		copy(x.Data[i*784:(i+1)*784], im.Pixels)
		if im.Label >= 0 && im.Label < 10 {
			target.Data[i*10+im.Label] = 1
		}
	}
	return runner.Sample{X: x, Target: target, Labels: labels}
}
