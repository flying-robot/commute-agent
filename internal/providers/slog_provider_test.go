package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// TestSlogProvider checks that each event is one JSON line with its level, its
// message, and its attributes, and that a debug event is not written.
func TestSlogProvider(t *testing.T) {
	var buf bytes.Buffer
	log := NewSlogProvider(&buf)
	log(t.Context(), internal.LevelDebug, "hidden")
	log(t.Context(), internal.LevelWarn, "tool failed", "step", 2)

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("lines = %q, want one", lines)
	}
	var event map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &event); err != nil {
		t.Fatal(err)
	}
	if event["level"] != "WARN" || event["msg"] != "tool failed" || event["step"] != 2.0 {
		t.Errorf("event = %v, want WARN, tool failed, step 2", event)
	}
}

// TestToSlogLevel checks the slog level of each level, and that an unknown
// level is info.
func TestToSlogLevel(t *testing.T) {
	tests := map[internal.Level]slog.Level{
		internal.LevelDebug: slog.LevelDebug,
		internal.LevelInfo:  slog.LevelInfo,
		internal.LevelWarn:  slog.LevelWarn,
		internal.LevelError: slog.LevelError,
		internal.Level(99):  slog.LevelInfo,
	}
	for level, want := range tests {
		if got := toSlogLevel(level); got != want {
			t.Errorf("toSlogLevel(%d) = %v, want %v", level, got, want)
		}
	}
}

// TestClockProviders checks that the clock gives the current time, that a sleep
// waits, and that a sleep stops at once with the error of a canceled context.
func TestClockProviders(t *testing.T) {
	before := time.Now()
	if now := NewClockProvider()(t.Context()); now.Before(before) || now.Sub(before) > time.Second {
		t.Errorf("now = %v, want about %v", now, before)
	}
	sleep := NewSleepProvider()
	if err := sleep(t.Context(), time.Millisecond); err != nil {
		t.Errorf("sleep = %v, want no error", err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := sleep(canceled, time.Hour); err == nil {
		t.Error("sleep with a canceled context: no error, want one")
	}
}
