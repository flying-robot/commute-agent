package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// chromaCollectionsURL is the URL of the collections of the default database of
// a Chroma server, from the address of the server.
const chromaCollectionsURL = "http://%s/api/v2/tenants/default_tenant/databases/default_database/collections"

// chromaMetadata is the metadata that the provider keeps with each document in
// Chroma. The times are Unix times in nanoseconds, because Chroma can compare
// numbers but not times. An Expires of zero means that the document never
// expires.
type chromaMetadata struct {
	Source  string `json:"source"`
	SavedAt int64  `json:"saved_at"`
	Expires int64  `json:"expires"`
}

// NewChromaProvider returns an internal.SaveFunc and an internal.SearchFunc that
// keep documents in a Chroma server at addr, such as localhost:8000. The
// provider embeds the texts itself, so that Chroma does not need a model. Each
// collection of the tools is a Chroma collection with cosine distance, which
// is made when it is first used.
func NewChromaProvider(addr string, embed internal.EmbedFunc) (internal.SaveFunc, internal.SearchFunc) {
	collectionsURL := fmt.Sprintf(chromaCollectionsURL, addr)

	// post sends body as JSON to the collections URL with path added, and
	// decodes the reply into out, if out is not nil.
	post := func(ctx context.Context, path string, body, out any) error {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, collectionsURL+path, bytes.NewReader(b))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return fmt.Errorf("failed to reach Chroma: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			msg, _ := io.ReadAll(resp.Body)
			return fmt.Errorf("chroma request failed with %s: %s", resp.Status, msg)
		}
		if out == nil {
			return nil
		}
		return json.NewDecoder(resp.Body).Decode(out)
	}

	// collectionPath returns the path of a collection under the collections URL,
	// and makes the collection if it does not exist.
	collectionPath := func(ctx context.Context, name string) (string, error) {
		var c struct {
			ID string `json:"id"`
		}
		err := post(ctx, "", map[string]any{
			"name":          name,
			"get_or_create": true,
			"configuration": map[string]any{"hnsw": map[string]any{"space": "cosine"}},
		}, &c)
		return "/" + c.ID, err
	}

	save := func(ctx context.Context, collection string, docs []internal.Document, ttl time.Duration) error {
		// First, embed the texts of the documents.
		if len(docs) == 0 {
			return nil
		}
		ids := make([]string, len(docs))
		texts := make([]string, len(docs))
		for i, d := range docs {
			ids[i] = d.ID
			texts[i] = d.Text
		}
		vectors, err := embed(ctx, texts, false)
		if err != nil {
			return err
		}
		if len(vectors) != len(docs) {
			return fmt.Errorf("got %d embeddings for %d documents", len(vectors), len(docs))
		}

		// Next, delete the expired documents of the collection, so that it does
		// not grow without limit.
		path, err := collectionPath(ctx, collection)
		if err != nil {
			return err
		}
		now := time.Now()
		err = post(ctx, path+"/delete", map[string]any{"where": map[string]any{"$and": []any{
			map[string]any{"expires": map[string]any{"$gt": 0}},
			map[string]any{"expires": map[string]any{"$lt": now.UnixNano()}},
		}}}, nil)
		if err != nil {
			return err
		}

		// Finally, add the documents, or replace the documents with the same IDs.
		var expires int64
		if ttl > 0 {
			expires = now.Add(ttl).UnixNano()
		}
		metadatas := make([]chromaMetadata, len(docs))
		for i, d := range docs {
			metadatas[i] = chromaMetadata{Source: d.Source, SavedAt: now.UnixNano(), Expires: expires}
		}
		return post(ctx, path+"/upsert", map[string]any{
			"ids":        ids,
			"embeddings": vectors,
			"documents":  texts,
			"metadatas":  metadatas,
		}, nil)
	}

	search := func(ctx context.Context, collection, source, query string, n int) ([]internal.Document, error) {
		// First, make a filter for the documents that have not expired, and only
		// for the source if it is set.
		path, err := collectionPath(ctx, collection)
		if err != nil {
			return nil, err
		}
		where := map[string]any{"$or": []any{
			map[string]any{"expires": 0},
			map[string]any{"expires": map[string]any{"$gt": time.Now().UnixNano()}},
		}}
		if source != "" {
			where = map[string]any{"$and": []any{where, map[string]any{"source": source}}}
		}

		// An empty query lists the documents. Chroma does not sort them, so get
		// all of them and return the n newest.
		if query == "" {
			var r struct {
				IDs       []string         `json:"ids"`
				Documents []string         `json:"documents"`
				Metadatas []chromaMetadata `json:"metadatas"`
			}
			err := post(ctx, path+"/get", map[string]any{"where": where, "include": []string{"documents", "metadatas"}}, &r)
			if err != nil {
				return nil, err
			}
			docs := chromaDocuments(r.IDs, r.Documents, r.Metadatas, nil)
			slices.SortFunc(docs, func(a, b internal.Document) int { return b.SavedAt.Compare(a.SavedAt) })
			return docs[:min(n, len(docs))], nil
		}

		// Otherwise, embed the query and get the n nearest documents, from the
		// nearest. Chroma gives one list of results for each query embedding,
		// and this search has one. Chroma takes no n larger than an unsigned
		// 32-bit integer, and the tools give math.MaxInt to get all documents.
		vectors, err := embed(ctx, []string{query}, true)
		if err != nil {
			return nil, err
		}
		if len(vectors) != 1 {
			return nil, fmt.Errorf("got %d embeddings for 1 query", len(vectors))
		}
		var r struct {
			IDs       [][]string         `json:"ids"`
			Documents [][]string         `json:"documents"`
			Metadatas [][]chromaMetadata `json:"metadatas"`
			Distances [][]float32        `json:"distances"`
		}
		err = post(ctx, path+"/query", map[string]any{
			"query_embeddings": vectors,
			"n_results":        min(n, math.MaxUint32),
			"where":            where,
			"include":          []string{"documents", "metadatas", "distances"},
		}, &r)
		if err != nil {
			return nil, err
		}
		if len(r.IDs) == 0 {
			return []internal.Document{}, nil
		}
		return chromaDocuments(r.IDs[0], r.Documents[0], r.Metadatas[0], r.Distances[0]), nil
	}

	return save, search
}

// chromaDocuments makes documents from the lists of a Chroma result. The cosine
// distance is 1 minus the cosine similarity, so the score of each document is 1
// minus its distance. Without distances, all scores are zero.
func chromaDocuments(ids, texts []string, metadatas []chromaMetadata, distances []float32) []internal.Document {
	docs := make([]internal.Document, len(ids))
	for i, id := range ids {
		docs[i] = internal.Document{
			ID:      id,
			Source:  metadatas[i].Source,
			Text:    texts[i],
			SavedAt: time.Unix(0, metadatas[i].SavedAt),
		}
		if distances != nil {
			docs[i].Score = 1 - distances[i]
		}
	}
	return docs
}
