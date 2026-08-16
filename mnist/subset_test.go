package mnist

import "testing"

func TestTakeBalanced(t *testing.T) {
	in := make([]Image, 0, 1000)
	for c := 0; c < 10; c++ {
		for i := 0; i < 100; i++ {
			in = append(in, Image{Label: c})
		}
	}
	got := TakeBalanced(in, 50, 1)
	if len(got) != 50 {
		t.Fatalf("len=%d", len(got))
	}
	counts := map[int]int{}
	for _, im := range got {
		counts[im.Label]++
	}
	for c := 0; c < 10; c++ {
		if counts[c] != 5 {
			t.Fatalf("class %d: %d want 5", c, counts[c])
		}
	}
	if len(TakeBalanced(in, 0, 1)) != len(in) {
		t.Fatal("n=0 should keep all")
	}
	if len(TakeBalanced(in, 5000, 1)) != len(in) {
		t.Fatal("n>len should keep all")
	}
	a := TakeBalanced(in, 80, 42)
	b := TakeBalanced(in, 80, 42)
	if a[0].Label != b[0].Label {
		t.Fatal("same seed should be deterministic")
	}
}
