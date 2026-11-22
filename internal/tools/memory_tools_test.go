package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// memoryStore is a fake store of facts for the memory tools. It keeps the facts
// by ID, and fail makes each call fail.
type memoryStore struct {
	facts map[string]string
	fail  error
}

func (m *memoryStore) save(_ context.Context, _ string, docs []internal.Document, _ time.Duration) error {
	if m.fail != nil {
		return m.fail
	}
	for _, d := range docs {
		m.facts[d.ID] = d.Text
	}
	return nil
}

func (m *memoryStore) search(_ context.Context, _, _, _ string, n int) ([]internal.Document, error) {
	if m.fail != nil {
		return nil, m.fail
	}
	var docs []internal.Document
	for id, text := range m.facts {
		if len(docs) < n {
			docs = append(docs, internal.Document{ID: id, Text: text})
		}
	}
	return docs, nil
}

func (m *memoryStore) del(_ context.Context, _, id string) (bool, error) {
	if m.fail != nil {
		return false, m.fail
	}
	_, ok := m.facts[id]
	delete(m.facts, id)
	return ok, nil
}

// run runs a tool with JSON arguments.
func run(t *testing.T, tool internal.Tool, args string) (string, error) {
	t.Helper()
	return tool.Run(t.Context(), json.RawMessage(args))
}

// TestSaveMemory checks that a fact is saved by its name in lower case, that a
// name and details are required, that the pickup plan cannot change, and that a
// failure of the store is an error.
func TestSaveMemory(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		fail    error
		wantErr string
		wantID  string
	}{
		{"saved", `{"name":"Trip to Work","details":"from A to B"}`, nil, "", "trip to work"},
		{"no details", `{"name":"work"}`, nil, "required", ""},
		{"pickup plan", `{"name":"Pickup","details":"contact: Mallory"}`, nil, "memory file", ""},
		{"store fails", `{"name":"work","details":"B"}`, errors.New("disk full"), "disk full", ""},
		{"not JSON", `{`, nil, "unmarshal", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &memoryStore{facts: map[string]string{}, fail: tt.fail}
			out, err := run(t, SaveMemory(m.save), tt.args)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("error = %v, want %q", err, tt.wantErr)
				}
				if len(m.facts) != 0 {
					t.Errorf("facts = %v, want none", m.facts)
				}
				return
			}
			if err != nil || !strings.Contains(out, tt.wantID) {
				t.Fatalf("result = %q, %v, want the name %q", out, err, tt.wantID)
			}
			if m.facts[tt.wantID] != tt.wantID+": from A to B" {
				t.Errorf("fact = %q, want it to start with its name", m.facts[tt.wantID])
			}
		})
	}
}

// TestForgetMemory checks that a fact is deleted by its name, that a missing
// fact and the pickup plan give an error, and that the pickup plan stays.
func TestForgetMemory(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		fail    error
		wantErr string
	}{
		{"deleted", `{"name":"Work"}`, nil, ""},
		{"no name", `{}`, nil, "required"},
		{"not saved", `{"name":"gym"}`, nil, "search_memory"},
		{"pickup plan", `{"name":"pickup"}`, nil, "memory file"},
		{"store fails", `{"name":"work"}`, errors.New("disk full"), "disk full"},
		{"not JSON", `{`, nil, "unmarshal"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &memoryStore{facts: map[string]string{"work": "work: B", internal.PickupFact: "contact: Bob"}, fail: tt.fail}
			_, err := run(t, ForgetMemory(m.del), tt.args)
			if tt.wantErr == "" {
				if _, ok := m.facts["work"]; err != nil || ok {
					t.Errorf("error = %v, work saved = %v, want it deleted", err, ok)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want %q", err, tt.wantErr)
			}
			if _, ok := m.facts[internal.PickupFact]; !ok {
				t.Error("the pickup plan was deleted")
			}
		})
	}
}

// TestSearchMemory checks that a search without a query gives all the facts,
// a search with a query gives at most memoryResults, and a failure of the store
// is an error.
func TestSearchMemory(t *testing.T) {
	m := &memoryStore{facts: map[string]string{}}
	for _, id := range []string{"a", "b", "c", "d", "e", "f", "g"} {
		m.facts[id] = id + ": fact"
	}
	count := func(args string) int {
		out, err := run(t, SearchMemory(m.search), args)
		if err != nil {
			t.Fatal(err)
		}
		var result struct {
			Memories []map[string]string `json:"memories"`
		}
		json.Unmarshal([]byte(out), &result)
		return len(result.Memories)
	}
	if n := count(`{}`); n != 7 {
		t.Errorf("facts without a query = %d, want 7", n)
	}
	if n := count(`{"query":"a"}`); n != memoryResults {
		t.Errorf("facts with a query = %d, want %d", n, memoryResults)
	}
	m.fail = errors.New("broken")
	if _, err := run(t, SearchMemory(m.search), `{}`); err == nil {
		t.Error("store fails: no error, want one")
	}
	if _, err := run(t, SearchMemory(m.search), `{`); err == nil {
		t.Error("not JSON: no error, want one")
	}
}
