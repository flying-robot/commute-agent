package providers

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// letterEmbed is a fake internal.EmbedFunc. The vector of a text has the count
// of each letter from a to z, so that texts with the same letters are near each
// other, and the result is the same in each run.
func letterEmbed(_ context.Context, texts []string, _ bool) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, 26)
		for _, r := range strings.ToLower(t) {
			if r >= 'a' && r <= 'z' {
				v[r-'a']++
			}
		}
		vectors[i] = v
	}
	return vectors, nil
}

// ids returns the IDs of the documents, in their order.
func ids(docs []internal.Document) []string {
	var out []string
	for _, d := range docs {
		out = append(out, d.ID)
	}
	return out
}

// TestVectorStore checks that a search finds the nearest documents of a source
// first, that an empty query lists the documents from the newest, that a save
// replaces a document with the same ID, and that a delete removes a document.
func TestVectorStore(t *testing.T) {
	ctx := t.Context()
	save, search, del := NewVectorStoreProvider(letterEmbed)

	if err := save(ctx, "c", nil, 0); err != nil {
		t.Fatalf("save of no documents: %v", err)
	}
	if err := save(ctx, "c", []internal.Document{
		{ID: "a", Source: "s1", Text: "aaaa"},
		{ID: "b", Source: "s1", Text: "bbbb"},
		{ID: "z", Source: "s2", Text: "aaab"},
	}, 0); err != nil {
		t.Fatal(err)
	}

	docs, err := search(ctx, "c", "", "aaaa", 10)
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(docs); !slices.Equal(got, []string{"a", "z", "b"}) {
		t.Errorf("search = %v, want a, z, b", got)
	}
	if docs[0].Score < 0.99 || docs[0].SavedAt.IsZero() {
		t.Errorf("first result = %+v, want a score of 1 and a save time", docs[0])
	}

	docs, _ = search(ctx, "c", "s1", "aaaa", 1)
	if got := ids(docs); !slices.Equal(got, []string{"a"}) {
		t.Errorf("search of source s1 with n 1 = %v, want a", got)
	}

	// The newest document is the one that was saved last.
	time.Sleep(time.Millisecond)
	save(ctx, "c", []internal.Document{{ID: "b", Source: "s1", Text: "aaaa"}}, 0)
	docs, _ = search(ctx, "c", "", "", 10)
	if len(docs) != 3 || docs[0].ID != "b" || docs[0].Score != 0 {
		t.Errorf("list = %v, want 3 documents with b first and no scores", ids(docs))
	}

	if ok, err := del(ctx, "c", "a"); !ok || err != nil {
		t.Errorf("delete a = %v, %v, want true", ok, err)
	}
	if ok, _ := del(ctx, "c", "a"); ok {
		t.Error("second delete of a = true, want false")
	}
	if docs, _ := search(ctx, "missing", "", "a", 10); len(docs) != 0 {
		t.Errorf("search of a missing collection = %v, want none", ids(docs))
	}
}

// TestVectorStoreExpires checks that a document with a ttl is not found after it
// expires, and that a document without a ttl stays.
func TestVectorStoreExpires(t *testing.T) {
	ctx := t.Context()
	save, search, del := NewVectorStoreProvider(letterEmbed)
	save(ctx, "c", []internal.Document{{ID: "old", Text: "a"}}, time.Millisecond)
	save(ctx, "c", []internal.Document{{ID: "kept", Text: "a"}}, 0)
	time.Sleep(5 * time.Millisecond)

	docs, _ := search(ctx, "c", "", "a", 10)
	if got := ids(docs); !slices.Equal(got, []string{"kept"}) {
		t.Errorf("search = %v, want only kept", got)
	}
	if ok, _ := del(ctx, "c", "old"); ok {
		t.Error("delete of an expired document = true, want false")
	}
}

// TestVectorStoreEmbedErrors checks that a failed embedding, or a wrong number
// of vectors, is an error of the save and of the search.
func TestVectorStoreEmbedErrors(t *testing.T) {
	tests := []struct {
		name  string
		embed internal.EmbedFunc
	}{
		{"embedding fails", func(context.Context, []string, bool) ([][]float32, error) {
			return nil, errors.New("broken")
		}},
		{"too many vectors", func(_ context.Context, texts []string, _ bool) ([][]float32, error) {
			return make([][]float32, len(texts)+1), nil
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			save, search, _ := NewVectorStoreProvider(tt.embed)
			if err := save(t.Context(), "c", []internal.Document{{ID: "a", Text: "a"}}, 0); err == nil {
				t.Error("save: no error, want one")
			}
			if _, err := search(t.Context(), "c", "", "a", 1); err == nil {
				t.Error("search: no error, want one")
			}
		})
	}
}

// TestUnitAndDot checks that unit makes a vector with a length of 1, keeps a
// zero vector, and that dot ignores the extra values of a longer vector.
func TestUnitAndDot(t *testing.T) {
	if got := unit([]float32{3, 4}); got[0] != 0.6 || got[1] != 0.8 {
		t.Errorf("unit(3, 4) = %v, want 0.6, 0.8", got)
	}
	if got := unit([]float32{0, 0}); got[0] != 0 || got[1] != 0 {
		t.Errorf("unit(0, 0) = %v, want zeros", got)
	}
	if got := dot([]float32{1, 2, 3}, []float32{4, 5}); got != 14 {
		t.Errorf("dot = %v, want 14", got)
	}
}
