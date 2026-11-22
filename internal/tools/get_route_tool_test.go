package tools

import (
	"encoding/json"
	"image"
	"image/color"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

// TestGetRouteArguments checks the errors of the arguments, which the tool
// finds before it sends a request. Valid arguments with no API key stop at the
// check of the key, so that the test needs no network.
func TestGetRouteArguments(t *testing.T) {
	future := time.Now().Add(time.Hour).Format(time.RFC3339)
	past := time.Now().Add(-time.Hour).Format(time.RFC3339)
	recent := time.Now().Add(-2 * time.Minute).Format(time.RFC3339)
	tests := []struct {
		name    string
		args    string
		wantErr string
	}{
		{"not JSON", `{`, "unmarshal"},
		{"no destination", `{"start":"A"}`, "required"},
		{"departure and arrival", `{"start":"A","destination":"B","departure":"` + future + `","arrival":"` + future + `"}`, "not both"},
		{"departure not RFC 3339", `{"start":"A","destination":"B","departure":"8am"}`, "RFC 3339"},
		{"departure in the past", `{"start":"A","destination":"B","departure":"` + past + `"}`, "in the past"},
		{"arrival not RFC 3339", `{"start":"A","destination":"B","arrival":"9am"}`, "RFC 3339"},
		{"arrival in the past", `{"start":"A","destination":"B","arrival":"` + past + `"}`, "in the past"},
		{"unknown road to avoid", `{"start":"A","destination":"B","avoid":"bridges"}`, "not one of"},
		{"leave now", `{"start":"A","destination":"B"}`, "TOMTOM_API_KEY"},
		{"departure a short time ago means now", `{"start":"A","destination":"B","departure":"` + recent + `"}`, "TOMTOM_API_KEY"},
		{"departure in the future", `{"start":"A","destination":"B","departure":"` + future + `","option":2}`, "TOMTOM_API_KEY"},
		{"arrival in the future", `{"start":"A","destination":"B","arrival":"` + future + `"}`, "TOMTOM_API_KEY"},
		{"known roads to avoid", `{"start":"A","destination":"B","avoid":"tollRoads, ferries,"}`, "TOMTOM_API_KEY"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := run(t, GetRoute(""), tt.args)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

// TestTown checks the town and the state of TomTom addresses.
func TestTown(t *testing.T) {
	for _, tt := range []struct{ address, want string }{
		{"3401 Richmond Hwy, Alexandria, VA 22305", "Alexandria, VA"},
		{"Union Station, Washington, DC", "Washington, DC"},
		{"Tokyo", ""},
	} {
		if got := town(tt.address); got != tt.want {
			t.Errorf("town(%q) = %q, want %q", tt.address, got, tt.want)
		}
	}
}

// TestVia checks that via gives the two roads with the longest distance, with
// no direction, and a street name for a road with no number.
func TestVia(t *testing.T) {
	var route any
	json.Unmarshal([]byte(`{"guidance":{"instructions":[
		{"routeOffsetInMeters":0,"street":"Main St"},
		{"routeOffsetInMeters":100,"roadNumbers":["I-95 S"]},
		{"routeOffsetInMeters":5100,"roadNumbers":["I-495 N"]},
		{"routeOffsetInMeters":7100,"roadNumbers":["I-95 N"]},
		{"routeOffsetInMeters":8100}
	]}}`), &route)
	if got, want := via(route), []string{"I-95", "I-495"}; !reflect.DeepEqual(got, want) {
		t.Errorf("via = %v, want %v", got, want)
	}
	if got := via(nil); len(got) != 0 {
		t.Errorf("via of no route = %v, want none", got)
	}
}

// TestUnits checks the conversions of the units and their rounding.
func TestUnits(t *testing.T) {
	if got := minutes(89); got != 1 {
		t.Errorf("minutes(89) = %d, want 1", got)
	}
	if got := minutes(90); got != 2 {
		t.Errorf("minutes(90) = %d, want 2", got)
	}
	if got := km(12345); got != 12.3 {
		t.Errorf("km(12345) = %v, want 12.3", got)
	}
	if got := miles(16093.44); got != 10 {
		t.Errorf("miles(16093.44) = %v, want 10", got)
	}
}

// TestParseTime checks that the name of the argument is in the error.
func TestParseTime(t *testing.T) {
	if _, err := parseTime("departure", "2026-11-16T08:00:00-05:00"); err != nil {
		t.Errorf("valid time: %v", err)
	}
	if _, err := parseTime("arrival", "tomorrow"); err == nil || !strings.Contains(err.Error(), "arrival") {
		t.Errorf("error = %v, want the name of the argument", err)
	}
}

// TestPrune checks that only the old values are deleted.
func TestPrune(t *testing.T) {
	m := map[string]stamped[int]{
		"old": {1, time.Now().Add(-2 * time.Hour)},
		"new": {2, time.Now()},
	}
	prune(m, time.Hour)
	if _, ok := m["old"]; ok || len(m) != 1 {
		t.Errorf("map = %v, want only the new value", m)
	}
}

// TestRouteMapsNotFound checks that an unknown map is not found, with no
// request for tiles.
func TestRouteMapsNotFound(t *testing.T) {
	w := httptest.NewRecorder()
	RouteMaps().ServeHTTP(w, httptest.NewRequest("GET", "/unknown.png", nil))
	if w.Code != 404 {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

// TestDraw checks that a line colors the pixels between its ends, and that a
// dot outside the image changes nothing.
func TestDraw(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	red := color.RGBA{255, 0, 0, 255}
	line(img, 1, 1, 8, 1, 0, red)
	for x := 1; x <= 8; x++ {
		if img.RGBAAt(x, 1) != red {
			t.Errorf("pixel %d,1 = %v, want red", x, img.RGBAAt(x, 1))
		}
	}
	if img.RGBAAt(1, 5) == red {
		t.Error("pixel 1,5 is red, want it unchanged")
	}
	dot(img, -50, -50, 2, red)
}
