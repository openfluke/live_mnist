// Package mnist downloads and splits the classic IDX MNIST dataset.
package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	baseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
)

// Image is a 28×28 grayscale sample.
type Image struct {
	Pixels []float32 // length 784, 0..1
	Label  int
}

// Split is an 80/20 partition of the official training set (plus test held aside).
type Split struct {
	Train []Image // 80% of train-images
	Val   []Image // 20% of train-images
	Test  []Image // official test set (optional eval)
}

// Load downloads (if needed) into dir and returns an 80/20 split of the training set.
func Load(dir string) (*Split, error) {
	if dir == "" {
		dir = "data"
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	files := []string{
		"train-images-idx3-ubyte.gz",
		"train-labels-idx1-ubyte.gz",
		"t10k-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz",
	}
	for _, f := range files {
		if err := ensure(dir, f); err != nil {
			return nil, err
		}
	}
	trainX, err := readImages(filepath.Join(dir, "train-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	trainY, err := readLabels(filepath.Join(dir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	if len(trainX) != len(trainY) {
		return nil, fmt.Errorf("mnist: train x/y mismatch %d/%d", len(trainX), len(trainY))
	}
	testX, err := readImages(filepath.Join(dir, "t10k-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	testY, err := readLabels(filepath.Join(dir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, err
	}

	all := make([]Image, len(trainX))
	for i := range trainX {
		all[i] = Image{Pixels: trainX[i], Label: int(trainY[i])}
	}
	// Deterministic 80/20 by index (every 5th → val).
	train, val := make([]Image, 0, len(all)*4/5), make([]Image, 0, len(all)/5)
	for i, im := range all {
		if i%5 == 0 {
			val = append(val, im)
		} else {
			train = append(train, im)
		}
	}
	test := make([]Image, len(testX))
	for i := range testX {
		test[i] = Image{Pixels: testX[i], Label: int(testY[i])}
	}
	return &Split{Train: train, Val: val, Test: test}, nil
}

func ensure(dir, name string) error {
	path := filepath.Join(dir, name)
	if st, err := os.Stat(path); err == nil && st.Size() > 0 {
		return nil
	}
	url := baseURL + name
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("mnist download %s: %w", name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("mnist download %s: HTTP %s", name, resp.Status)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func readImages(path string) ([][]float32, error) {
	r, err := openGZ(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var magic, n, rows, cols int32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 0x00000803 {
		return nil, fmt.Errorf("mnist: bad image magic %x", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.BigEndian, &rows); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.BigEndian, &cols); err != nil {
		return nil, err
	}
	pix := int(rows * cols)
	out := make([][]float32, n)
	buf := make([]byte, pix)
	for i := 0; i < int(n); i++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		im := make([]float32, pix)
		for j, b := range buf {
			im[j] = float32(b) / 255
		}
		out[i] = im
	}
	return out, nil
}

func readLabels(path string) ([]byte, error) {
	r, err := openGZ(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var magic, n int32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 0x00000801 {
		return nil, fmt.Errorf("mnist: bad label magic %x", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	out := make([]byte, n)
	if _, err := io.ReadFull(r, out); err != nil {
		return nil, err
	}
	return out, nil
}

func openGZ(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	return &gzFile{gz: gz, f: f}, nil
}

type gzFile struct {
	gz *gzip.Reader
	f  *os.File
}

func (g *gzFile) Read(p []byte) (int, error) { return g.gz.Read(p) }
func (g *gzFile) Close() error {
	err1 := g.gz.Close()
	err2 := g.f.Close()
	if err1 != nil {
		return err1
	}
	return err2
}
