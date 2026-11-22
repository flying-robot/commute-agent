package providers

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/flying-robot/commute-agent/internal"
)

// TestNewNtfyProvider checks that the provider needs a topic and a known
// failure.
func TestNewNtfyProvider(t *testing.T) {
	if _, err := NewNtfyProvider("", ""); err == nil {
		t.Error("no topic: no error, want one")
	}
	if _, err := NewNtfyProvider("topic", "rejected"); err == nil {
		t.Error("unknown failure: no error, want one")
	}
	if _, err := NewNtfyProvider("topic", ""); err != nil {
		t.Errorf("topic and no failure: %v, want no error", err)
	}
}

// TestNtfyFailures checks each injected failure without a publish. The timeout
// gets a context that is already canceled, so that the test does not wait, and
// it must return the error of the context.
func TestNtfyFailures(t *testing.T) {
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	tests := []struct {
		fail string
		ctx  context.Context
		want string
	}{
		{"error", t.Context(), "status 500"},
		{"rate_limit", t.Context(), "status 429"},
		{"timeout", canceled, context.Canceled.Error()},
	}
	for _, tt := range tests {
		t.Run(tt.fail, func(t *testing.T) {
			notify, err := NewNtfyProvider("topic", tt.fail)
			if err != nil {
				t.Fatal(err)
			}
			id, err := notify(tt.ctx, "Bob", "Pickup request", "body")
			if err == nil || id != "" || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("notify = %q, %v, want an error with %q", id, err, tt.want)
			}
			if errors.Is(err, internal.ErrNoResponse) {
				t.Errorf("error %v has ErrNoResponse, want not", err)
			}
		})
	}
}
