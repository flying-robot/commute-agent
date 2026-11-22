package providers

import (
	"testing"
	"time"
)

// TestChromaDocuments checks that the score is 1 minus the cosine distance, and
// that all scores are zero without distances.
func TestChromaDocuments(t *testing.T) {
	saved := time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)
	metadatas := []chromaMetadata{{Source: "s", SavedAt: saved.UnixNano()}}
	docs := chromaDocuments([]string{"a"}, []string{"text"}, metadatas, []float32{0.25})
	if len(docs) != 1 || docs[0].ID != "a" || docs[0].Text != "text" || docs[0].Source != "s" || !docs[0].SavedAt.Equal(saved) || docs[0].Score != 0.75 {
		t.Errorf("documents = %+v", docs)
	}
	if docs := chromaDocuments([]string{"a"}, []string{"text"}, metadatas, nil); docs[0].Score != 0 {
		t.Errorf("score without distances = %v, want 0", docs[0].Score)
	}
}
