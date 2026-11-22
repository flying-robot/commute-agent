package providers

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// storedDocument is a document in the vector store, with its unit vector and the
// time that it expires. A zero expiry time means that it never expires.
type storedDocument struct {
	doc     internal.Document
	vector  []float32
	expires time.Time
}

// NewVectorStoreProvider returns an internal.SaveFunc, an internal.SearchFunc,
// and an internal.DeleteFunc that share one vector store in memory. A search
// compares the query with each document of the collection, which is fast enough
// for the few documents of the file store and of the eval. The store is lost
// when the server stops.
func NewVectorStoreProvider(embed internal.EmbedFunc) (internal.SaveFunc, internal.SearchFunc, internal.DeleteFunc) {
	// The mutex and the collections are captured by both functions, so that a
	// search finds what a save added.
	var mu sync.Mutex
	collections := map[string]map[string]storedDocument{}

	save := func(ctx context.Context, collection string, docs []internal.Document, ttl time.Duration) error {
		// First, embed the texts of the documents. This happens before the lock,
		// because it is slow, and it does not use the store.
		if len(docs) == 0 {
			return nil
		}
		texts := make([]string, len(docs))
		for i, d := range docs {
			texts[i] = d.Text
		}
		vectors, err := embed(ctx, texts, false)
		if err != nil {
			return err
		}
		if len(vectors) != len(docs) {
			return fmt.Errorf("got %d embeddings for %d documents", len(vectors), len(docs))
		}

		// Finally, store each document with its vector and its expiry time, and
		// delete the expired documents, so that the store does not grow without
		// limit.
		now := time.Now()
		var expires time.Time
		if ttl > 0 {
			expires = now.Add(ttl)
		}
		mu.Lock()
		defer mu.Unlock()
		c := collections[collection]
		if c == nil {
			c = map[string]storedDocument{}
			collections[collection] = c
		}
		deleteExpired(c, now)
		for i, d := range docs {
			d.SavedAt = now
			c[d.ID] = storedDocument{doc: d, vector: unit(vectors[i]), expires: expires}
		}
		return nil
	}

	search := func(ctx context.Context, collection, source, query string, n int) ([]internal.Document, error) {
		// First, embed the query. An empty query lists the documents, so it
		// needs no embedding.
		var q []float32
		if query != "" {
			vectors, err := embed(ctx, []string{query}, true)
			if err != nil {
				return nil, err
			}
			if len(vectors) != 1 {
				return nil, fmt.Errorf("got %d embeddings for 1 query", len(vectors))
			}
			q = unit(vectors[0])
		}

		// Next, delete the expired documents, so that a search never finds old
		// data, and score each other document of the source. The vectors are
		// unit vectors, so their dot product is the cosine similarity.
		type scored struct {
			doc   internal.Document
			score float32
		}
		mu.Lock()
		c := collections[collection]
		deleteExpired(c, time.Now())
		var results []scored
		for _, s := range c {
			if source == "" || s.doc.Source == source {
				results = append(results, scored{s.doc, dot(s.vector, q)})
			}
		}
		mu.Unlock()

		// Finally, return the n documents with the highest scores, each with its
		// score, or the n newest documents for an empty query. All scores are
		// zero then, because dot gets no query vector.
		slices.SortFunc(results, func(a, b scored) int {
			if query == "" {
				return b.doc.SavedAt.Compare(a.doc.SavedAt)
			}
			return cmp.Compare(b.score, a.score)
		})
		docs := []internal.Document{}
		for _, r := range results[:min(n, len(results))] {
			r.doc.Score = r.score
			docs = append(docs, r.doc)
		}
		return docs, nil
	}

	del := func(_ context.Context, collection, id string) (bool, error) {
		// An expired document counts as deleted already, so it is deleted first.
		mu.Lock()
		defer mu.Unlock()
		c := collections[collection]
		deleteExpired(c, time.Now())
		_, ok := c[id]
		delete(c, id)
		return ok, nil
	}

	return save, search, del
}

// deleteExpired deletes the documents of a collection that expired before now.
func deleteExpired(c map[string]storedDocument, now time.Time) {
	maps.DeleteFunc(c, func(_ string, s storedDocument) bool {
		return !s.expires.IsZero() && now.After(s.expires)
	})
}

// unit returns v scaled to a length of 1, so that the dot product of two unit
// vectors is their cosine similarity. A zero vector stays zero.
func unit(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	if sum == 0 {
		return out
	}
	norm := float32(math.Sqrt(sum))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// dot returns the dot product of two vectors of the same length.
func dot(a, b []float32) float32 {
	var sum float32
	for i := range min(len(a), len(b)) {
		sum += a[i] * b[i]
	}
	return sum
}
