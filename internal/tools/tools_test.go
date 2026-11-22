package tools

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// TestDecode checks that no arguments give an empty map, and that arguments
// that are not a JSON object give an error.
func TestDecode(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		want    map[string]any
		wantErr bool
	}{
		{"no arguments", "", map[string]any{}, false},
		{"object", `{"a":1}`, map[string]any{"a": 1.0}, false},
		{"not an object", `[1]`, nil, true},
		{"not JSON", `{`, nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := decode(json.RawMessage(tt.args))
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, want an error: %v", err, tt.wantErr)
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("decode = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestToJSON checks that a value becomes its JSON text.
func TestToJSON(t *testing.T) {
	got, err := toJSON(map[string]int{"a": 1})
	if err != nil || got != `{"a":1}` {
		t.Errorf("toJSON = %q, %v, want {\"a\":1}", got, err)
	}
}

// TestGet checks that get follows keys and indexes, and gives the zero value
// for a path that does not exist or a value of a different type.
func TestGet(t *testing.T) {
	var v any
	json.Unmarshal([]byte(`{"a":{"b":[{"c":"x"},2]}}`), &v)
	if got := get[string](v, "a", "b", 0, "c"); got != "x" {
		t.Errorf("get a.b.0.c = %q, want x", got)
	}
	if got := get[float64](v, "a", "b", 1); got != 2 {
		t.Errorf("get a.b.1 = %v, want 2", got)
	}
	if got := get[string](v, "a", "b", 5); got != "" {
		t.Errorf("get of a missing index = %q, want empty", got)
	}
	if got := get[string](v, "a", "missing"); got != "" {
		t.Errorf("get of a missing key = %q, want empty", got)
	}
	if got := get[string](v, "a", "b", 1); got != "" {
		t.Errorf("get of a number as a string = %q, want empty", got)
	}
}

// TestFetch checks that fetch sends the user agent and the Accept header, gives
// the body, sends a request again once after a server error or too many
// requests, and does not send it again after a different error.
func TestFetch(t *testing.T) {
	tests := []struct {
		name      string
		statuses  []int
		wantCalls int32
		wantErr   string
	}{
		{"success", []int{200}, 1, ""},
		{"server error, then success", []int{500, 200}, 2, ""},
		{"too many requests, then success", []int{429, 200}, 2, ""},
		{"server error two times", []int{503, 503}, 2, "status 503"},
		{"not found is not sent again", []int{404}, 1, "status 404"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				n := calls.Add(1)
				if r.Header.Get("User-Agent") != userAgent || r.Header.Get("Accept") != "text/plain" {
					t.Errorf("headers = %v, want the user agent and the Accept header", r.Header)
				}
				w.WriteHeader(tt.statuses[min(int(n), len(tt.statuses))-1])
				w.Write([]byte("body"))
			}))
			defer srv.Close()

			got, err := fetch(t.Context(), srv.URL, "text/plain")
			if calls.Load() != tt.wantCalls {
				t.Errorf("calls = %d, want %d", calls.Load(), tt.wantCalls)
			}
			switch {
			case tt.wantErr == "" && (err != nil || string(got) != "body"):
				t.Errorf("fetch = %q, %v, want body", got, err)
			case tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)):
				t.Errorf("error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

// TestFetchCanceled checks that a canceled context stops the wait before the
// retry.
func TestFetchCanceled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := fetch(ctx, srv.URL, ""); err == nil {
		t.Error("fetch with a canceled context: no error, want one")
	}
}

// TestAll checks that each agent gets its tools.
func TestAll(t *testing.T) {
	planner, conditions := All("", nil, nil, time.Minute, 5, nil, nil, nil)
	names := func(tools []internal.Tool) []string {
		var n []string
		for _, tool := range tools {
			n = append(n, tool.Name)
		}
		return n
	}
	if got := names(conditions); !reflect.DeepEqual(got, []string{"get_route_weather", "get_current_news"}) {
		t.Errorf("conditions tools = %v", got)
	}
	if got := names(planner); len(got) != 7 || got[0] != "get_current_time" || got[1] != "get_route" {
		t.Errorf("planner tools = %v", got)
	}
}
