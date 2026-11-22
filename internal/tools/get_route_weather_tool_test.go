package tools

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// TestGetRouteWeatherArguments checks the errors of the arguments, which the
// tool finds before it sends a request.
func TestGetRouteWeatherArguments(t *testing.T) {
	tool := GetRouteWeather("", nil, nil, time.Minute)
	for _, tt := range []struct{ name, args, wantErr string }{
		{"not JSON", `{`, "unmarshal"},
		{"no destination", `{"start":"A"}`, "required"},
		{"departure not RFC 3339", `{"start":"A","destination":"B","departure":"8am"}`, "departure"},
		{"arrival not RFC 3339", `{"start":"A","destination":"B","departure":"2026-11-16T08:00:00-05:00","arrival":"9am"}`, "arrival"},
		{"no API key", `{"start":"A","destination":"B","departure":"2026-11-16T08:00:00-05:00","arrival":"2026-11-16T09:00:00-05:00"}`, "TOMTOM_API_KEY"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := run(t, tool, tt.args); err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

// TestGetRouteWeatherSaved checks a full run with no network: the places are
// already geocoded, and the weather of each place is saved, so the result gives
// the saved weather and its age. A saved text that is not JSON, and a failure
// of the store, are errors.
func TestGetRouteWeatherSaved(t *testing.T) {
	places.Lock()
	places.byQuery["test start"] = stamped[[2]string]{[2]string{"Start, MD", "1.0,2.0"}, time.Now()}
	places.byQuery["test end"] = stamped[[2]string]{[2]string{"End, MD", "3.0,4.0"}, time.Now()}
	places.Unlock()
	args := `{"start":"Test Start","destination":"Test End","departure":"2026-11-16T08:00:00-05:00","arrival":"2026-11-16T09:00:00-05:00"}`

	text := `{"place":"P","time":"T","weather":{"forecast":"Rain","alerts":[]}}`
	search := func(_ context.Context, collection, source, query string, _ int) ([]internal.Document, error) {
		if collection != weatherCollection || !strings.Contains(source, "@2026-11-16T") {
			t.Errorf("collection = %q, source = %q, want weather by position and hour", collection, source)
		}
		return []internal.Document{{Text: text, SavedAt: time.Now().Add(-4 * time.Minute)}}, nil
	}
	out, err := run(t, GetRouteWeather("key", nil, search, time.Minute), args)
	if err != nil || !strings.Contains(out, `"forecast":"Rain"`) || !strings.Contains(out, `"saved_minutes_ago":4`) {
		t.Errorf("result = %q, %v, want the saved weather and its age", out, err)
	}

	text = "not JSON"
	if _, err := run(t, GetRouteWeather("key", nil, search, time.Minute), args); err == nil {
		t.Error("saved text is not JSON: no error, want one")
	}
	failing := func(context.Context, string, string, string, int) ([]internal.Document, error) {
		return nil, errors.New("down")
	}
	if _, err := run(t, GetRouteWeather("key", nil, failing, time.Minute), args); err == nil {
		t.Error("store fails: no error, want one")
	}
}
