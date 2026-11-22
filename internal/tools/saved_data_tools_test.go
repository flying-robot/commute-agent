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

// TestGetCurrentTime checks the form of the result, because the tool reads the
// real clock: a time in RFC 3339, the time zone, and seven days from today. A
// time zone that does not exist is an error.
func TestGetCurrentTime(t *testing.T) {
	out, err := run(t, GetCurrentTime(), `{"timezone":"Asia/Tokyo"}`)
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Time     string   `json:"time"`
		Timezone string   `json:"timezone"`
		Weekday  string   `json:"weekday"`
		NextDays []string `json:"next_days"`
	}
	if err := json.Unmarshal([]byte(out), &result); err != nil {
		t.Fatal(err)
	}
	if _, err := time.Parse(time.RFC3339, result.Time); err != nil {
		t.Errorf("time = %q, want RFC 3339", result.Time)
	}
	if result.Timezone != "Asia/Tokyo" || result.Weekday == "" {
		t.Errorf("timezone = %q, weekday = %q, want Asia/Tokyo and a weekday", result.Timezone, result.Weekday)
	}
	if len(result.NextDays) != 7 || !strings.HasSuffix(result.NextDays[0], "(today)") || !strings.HasSuffix(result.NextDays[1], "(tomorrow)") {
		t.Errorf("next days = %v, want seven from today", result.NextDays)
	}
	if !strings.Contains(result.NextDays[0], "UTC offset +09:00") {
		t.Errorf("today = %q, want the offset of Tokyo", result.NextDays[0])
	}

	if _, err := run(t, GetCurrentTime(), `{}`); err != nil {
		t.Errorf("local time: %v", err)
	}
	if _, err := run(t, GetCurrentTime(), `{"timezone":"Mars/Base"}`); err == nil {
		t.Error("unknown time zone: no error, want one")
	}
	if _, err := run(t, GetCurrentTime(), `{`); err == nil {
		t.Error("not JSON: no error, want one")
	}
}

// savedSearch returns a fake SearchFunc that gives docs for each collection, or
// fails with err.
func savedSearch(docs map[string][]internal.Document, err error) internal.SearchFunc {
	return func(_ context.Context, collection, _, _ string, n int) ([]internal.Document, error) {
		if err != nil {
			return nil, err
		}
		d := docs[collection]
		return d[:min(n, len(d))], nil
	}
}

// TestSearchSavedData checks that the results have their age, that an empty
// result names the tool that gets new data, and the errors of the arguments and
// of the store.
func TestSearchSavedData(t *testing.T) {
	docs := map[string][]internal.Document{
		newsCollection: {{Text: "Title: Crash on I-495", SavedAt: time.Now().Add(-3 * time.Minute)}},
	}
	tool := SearchSavedData(savedSearch(docs, nil))

	out, err := run(t, tool, `{"collection":"news","query":"I-495"}`)
	if err != nil || !strings.Contains(out, `"checked_minutes_ago":3`) || !strings.Contains(out, "Crash on I-495") {
		t.Errorf("news = %q, %v, want the article with its age", out, err)
	}
	out, err = run(t, tool, `{"collection":"weather","query":"Bethesda"}`)
	if err != nil || !strings.Contains(out, "get_route_weather") {
		t.Errorf("empty weather = %q, %v, want the tool that gets new data", out, err)
	}

	for _, args := range []string{`{"collection":"sports","query":"x"}`, `{"collection":"news"}`, `{`} {
		if _, err := run(t, tool, args); err == nil {
			t.Errorf("args %s: no error, want one", args)
		}
	}
	if _, err := run(t, SearchSavedData(savedSearch(nil, errors.New("down"))), `{"collection":"news","query":"x"}`); err == nil {
		t.Error("store fails: no error, want one")
	}
}

// TestShowSavedData checks that the list shows each collection with a short
// line for each document, or one collection only, and the errors of the
// arguments and of the store.
func TestShowSavedData(t *testing.T) {
	weather := `{"place":"Bethesda, MD","time":"2026-11-16T08:00:00-05:00","weather":{"forecast":"Rain","temperature_f":48,"alerts":["Flood Watch"]}}`
	docs := map[string][]internal.Document{
		newsCollection:    {{Text: "Title: Crash on I-495\nSummary: long text", SavedAt: time.Now()}},
		weatherCollection: {{Text: weather, SavedAt: time.Now().Add(-2 * time.Minute)}},
	}
	out, err := run(t, ShowSavedData(savedSearch(docs, nil)), `{}`)
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Markdown string `json:"markdown"`
	}
	json.Unmarshal([]byte(out), &result)
	for _, want := range []string{
		"**Saved news** (1)",
		"- less than 1 minute ago: Crash on I-495",
		"**Saved weather** (1)",
		"- 2 minutes ago: Bethesda, MD at 2026-11-16T08:00:00-05:00: Rain, 48°F, 1 alerts",
	} {
		if !strings.Contains(result.Markdown, want) {
			t.Errorf("markdown =\n%s\nwant %q", result.Markdown, want)
		}
	}

	out, err = run(t, ShowSavedData(savedSearch(nil, nil)), `{"collection":"news"}`)
	if err != nil || !strings.Contains(out, "Nothing is saved.") || strings.Contains(out, "weather") {
		t.Errorf("news only = %q, %v, want an empty news list", out, err)
	}
	if _, err := run(t, ShowSavedData(savedSearch(nil, nil)), `{"collection":"sports"}`); err == nil {
		t.Error("unknown collection: no error, want one")
	}
	if _, err := run(t, ShowSavedData(savedSearch(nil, errors.New("down"))), `{}`); err == nil {
		t.Error("store fails: no error, want one")
	}
	if _, err := run(t, ShowSavedData(savedSearch(nil, nil)), `{`); err == nil {
		t.Error("not JSON: no error, want one")
	}
}

// TestAgo checks the text of each age.
func TestAgo(t *testing.T) {
	for _, tt := range []struct {
		age  time.Duration
		want string
	}{
		{10 * time.Second, "less than 1 minute ago"},
		{90 * time.Second, "1 minute ago"},
		{5 * time.Minute, "5 minutes ago"},
	} {
		if got := ago(time.Now().Add(-tt.age)); got != tt.want {
			t.Errorf("ago(%v) = %q, want %q", tt.age, got, tt.want)
		}
	}
}

// TestSummary checks that a weather document without a forecast says so, and
// that a long line is cut.
func TestSummary(t *testing.T) {
	if got := summary(`{"place":"Tokyo","time":"t","weather":null}`); got != "Tokyo at t: no forecast" {
		t.Errorf("summary = %q, want no forecast", got)
	}
	long := strings.Repeat("a", shownLength+10)
	if got := summary(long); got != strings.Repeat("a", shownLength)+"..." {
		t.Errorf("summary of a long line has %d characters, want %d", len(got), shownLength+3)
	}
}
