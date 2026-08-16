package mnist

import "math/rand/v2"

const nClass = 10

// TakeBalanced returns at most n examples, spreading as evenly as possible
// across digit classes. n<=0 or n>=len(in) returns in unchanged.
// The subset is deterministic for a given seed so resume offsets stay valid.
func TakeBalanced(in []Image, n int, seed uint64) []Image {
	if n <= 0 || n >= len(in) {
		return in
	}
	by := make([][]int, nClass)
	for i, im := range in {
		l := im.Label
		if l < 0 || l >= nClass {
			continue
		}
		by[l] = append(by[l], i)
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x51B5E7))
	for c := range by {
		rng.Shuffle(len(by[c]), func(i, j int) { by[c][i], by[c][j] = by[c][j], by[c][i] })
	}
	per := n / nClass
	extra := n % nClass
	out := make([]Image, 0, n)
	for c := 0; c < nClass; c++ {
		take := per
		if c < extra {
			take++
		}
		if take > len(by[c]) {
			take = len(by[c])
		}
		for i := 0; i < take; i++ {
			out = append(out, in[by[c][i]])
		}
	}
	return out
}
