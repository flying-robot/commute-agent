package providers

import (
	"context"
	"encoding/json"
	"errors"
	"html/template"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/flying-robot/commute-agent/internal"
	"github.com/labstack/echo/v5"
)

// TestSecondsAndIndent checks the format of a duration, and that indent changes
// only JSON.
func TestSecondsAndIndent(t *testing.T) {
	if got := seconds(2340 * time.Millisecond); got != "2.3 s" {
		t.Errorf("seconds = %q, want 2.3 s", got)
	}
	if got := indent(`{"a":1}`); got != "{\n  \"a\": 1\n}" {
		t.Errorf("indent of JSON = %q", got)
	}
	if got := indent("tool failed: broken"); got != "tool failed: broken" {
		t.Errorf("indent of text = %q, want no change", got)
	}
}

// TestSavedNotice checks the notice for the age of the oldest saved data, and
// that a turn without saved data gets no notice.
func TestSavedNotice(t *testing.T) {
	result := func(content string) internal.Message { return internal.Message{Role: "tool", Content: content} }
	tests := []struct {
		name string
		turn []internal.Message
		want string
	}{
		{"no saved data", []internal.Message{result(`{"a":1}`), result("tool failed: x")}, ""},
		{"now", []internal.Message{result(`{"saved_minutes_ago":0}`)}, "less than a minute ago"},
		{"one minute", []internal.Message{result(`{"saved_minutes_ago":1}`)}, "1 minute ago"},
		{"oldest of two", []internal.Message{result(`{"saved_minutes_ago":2}`), result(`{"saved_minutes_ago":7}`)}, "7 minutes ago"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := savedNotice(tt.turn)
			if (tt.want == "") != (got == "") || !strings.Contains(got, tt.want) {
				t.Errorf("notice = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestWithMentions checks that the mention text of a tool result is added once,
// and not when the answer has it already.
func TestWithMentions(t *testing.T) {
	turn := []internal.Message{
		{Role: "tool", Content: `{"mention":"**News:** none."}`},
		{Role: "user", Content: `{"mention":"not a tool"}`},
	}
	if got := withMentions(turn, "Answer"); got != "Answer\n\n**News:** none." {
		t.Errorf("withMentions = %q", got)
	}
	if got := withMentions(turn, "Answer **News:** none."); got != "Answer **News:** none." {
		t.Errorf("withMentions with the mention = %q, want no change", got)
	}
}

// TestNewProgressView checks the step event of a reply with tool calls, and of
// a tool result that failed.
func TestNewProgressView(t *testing.T) {
	reply := internal.Message{Role: "assistant", Duration: time.Second, ToolCalls: []internal.ToolCall{{Name: "a"}, {Name: "b"}}}
	if got := newProgressView(2, reply); got.Step != 2 || got.Calls != "a, b" || got.Duration != "1.0 s" {
		t.Errorf("view of a reply = %+v", got)
	}
	failed := internal.Message{Role: "tool", ToolName: "a", Content: internal.ToolFailed + "x"}
	if got := newProgressView(2, failed); got.Tool != "a" || !got.Failed {
		t.Errorf("view of a failed result = %+v", got)
	}
}

// TestNewReplyView checks that each reply is a step, that each tool result
// belongs to its call in order, and that the total time adds all the times.
func TestNewReplyView(t *testing.T) {
	turn := []internal.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Duration: time.Second, Thinking: "hmm", ToolCalls: []internal.ToolCall{
			{Name: "a", Arguments: json.RawMessage(`{"x":1}`)},
			{Name: "b"},
		}},
		{Role: "tool", ToolName: "a", Content: `{"ok":true}`, Duration: time.Second},
		{Role: "tool", ToolName: "b", Content: internal.ToolFailed + "broken"},
		{Role: "tool", ToolName: "extra", Content: "no call"},
		{Role: "assistant", Content: "answer", Duration: time.Second},
	}
	view := newReplyView(turn, "answer", false)
	if len(view.Steps) != 2 || len(view.Calls) != 2 || view.Total != "3.0 s" {
		t.Fatalf("view = %+v, want 2 steps, 2 calls, 3.0 s", view)
	}
	a, b := view.Calls[0], view.Calls[1]
	if a.Result != "{\n  \"ok\": true\n}" || a.Failed || a.Arguments != "{\n  \"x\": 1\n}" || !b.Failed {
		t.Errorf("calls = %+v, %+v", a, b)
	}
	if view.Steps[0].Thinking != "hmm" || view.Steps[1].Content != "answer" || view.Steps[1].Number != 2 {
		t.Errorf("steps = %+v", view.Steps)
	}
}

// TestRender checks the HTML of the templates, and that a template error shows
// as escaped text.
func TestRender(t *testing.T) {
	reply := render(replyTemplate, newReplyView([]internal.Message{
		{Role: "assistant", ToolCalls: []internal.ToolCall{{Name: "get_route"}}},
		{Role: "tool", Content: internal.ToolFailed + "<broken>"},
	}, "Answer", false))
	for _, want := range []string{`class="msg assistant"`, "get_route", "&lt;broken&gt;", "steps: 1"} {
		if !strings.Contains(reply, want) {
			t.Errorf("reply HTML does not have %q", want)
		}
	}
	if failed := render(replyTemplate, replyView{Answer: "x", Failed: true}); !strings.Contains(failed, `class="msg error"`) {
		t.Errorf("failed reply HTML = %q", failed)
	}

	alerts := render(alertsTemplate, []alertView{
		{Alert: internal.Alert{ID: "1", Kind: "feedback", Text: "How long?", Status: "pending"}},
		{Alert: internal.Alert{ID: "2", Kind: "feedback", Text: "Thank you.", Status: "answered"}, Error: "bad"},
	})
	if !strings.Contains(alerts, `hx-post="/alerts/1/feedback"`) || strings.Contains(alerts, `hx-post="/alerts/2/feedback"`) || !strings.Contains(alerts, `<div class="alert-error">bad</div>`) {
		t.Errorf("alerts HTML = %q", alerts)
	}

	broken := template.Must(template.New("x").Parse("{{.Missing}}"))
	if got := render(broken, struct{}{}); !strings.Contains(got, "can&#39;t evaluate field Missing") {
		t.Errorf("render error = %q", got)
	}
}

// newTestContext returns an echo context for a request, with the HTMX header if
// htmx is true, and its recorder.
func newTestContext(method string, htmx bool) (*echo.Context, *httptest.ResponseRecorder) {
	req := httptest.NewRequest(method, "/", nil)
	if htmx {
		req.Header.Set("HX-Request", "true")
	}
	rec := httptest.NewRecorder()
	return echo.New().NewContext(req, rec), rec
}

// TestReplyError checks that an error is HTML for the chat page and an HTTP
// error for other callers.
func TestReplyError(t *testing.T) {
	c, rec := newTestContext(http.MethodPost, true)
	if err := replyError(c, http.StatusBadRequest, "<bad>"); err != nil || rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "&lt;bad&gt;") {
		t.Errorf("HTMX error = %v, %d, %q", err, rec.Code, rec.Body)
	}
	c, _ = newTestContext(http.MethodPost, false)
	if err := replyError(c, http.StatusBadRequest, "bad"); err == nil {
		t.Error("JSON error: no error, want one")
	}
}

// TestReplyAlert checks the reply to an answer: JSON for other callers, and for
// the chat page the alert with the error, as the agent has it now.
func TestReplyAlert(t *testing.T) {
	alert := internal.Alert{ID: "1", Kind: "feedback", Text: "How long?", Status: "pending"}
	agent := internal.Agent{Alerts: func(context.Context) []internal.Alert { return []internal.Alert{alert} }}

	c, rec := newTestContext(http.MethodPost, false)
	if err := replyAlert(c, agent, alert, nil); err != nil || !strings.Contains(rec.Body.String(), `"id":"1"`) {
		t.Errorf("JSON reply = %v, %q", err, rec.Body)
	}
	c, _ = newTestContext(http.MethodPost, false)
	if err := replyAlert(c, agent, internal.Alert{}, errors.New("bad")); err == nil {
		t.Error("JSON reply to an error: no error, want one")
	}

	c, rec = newTestContext(http.MethodPost, true)
	c.SetPathValues(echo.PathValues{{Name: "id", Value: "1"}})
	if err := replyAlert(c, agent, internal.Alert{}, errors.New("give a number")); err != nil {
		t.Fatal(err)
	}
	if body := rec.Body.String(); !strings.Contains(body, `id="alert-1"`) || !strings.Contains(body, "How long?") || !strings.Contains(body, "give a number") {
		t.Errorf("HTMX reply to an error = %q", body)
	}
}

// TestConversationID checks that a browser with a cookie keeps its conversation,
// and that a browser without one gets a new conversation in a cookie.
func TestConversationID(t *testing.T) {
	c, _ := newTestContext(http.MethodPost, false)
	c.Request().AddCookie(&http.Cookie{Name: conversationCookie, Value: "abc"})
	if got := conversationID(c); got != "abc" {
		t.Errorf("conversation = %q, want abc", got)
	}

	c, rec := newTestContext(http.MethodPost, false)
	id := conversationID(c)
	cookie := rec.Result().Cookies()
	if id == "" || len(cookie) != 1 || cookie[0].Value != id || !cookie[0].HttpOnly {
		t.Errorf("new conversation = %q, cookies = %v", id, cookie)
	}
}

// TestWriteEvent checks that each line of the data has its own data field.
func TestWriteEvent(t *testing.T) {
	rec := httptest.NewRecorder()
	writeEvent(rec, "step", "a\nb")
	if got := rec.Body.String(); got != "event: step\ndata: a\ndata: b\n\n" {
		t.Errorf("event = %q", got)
	}
}
