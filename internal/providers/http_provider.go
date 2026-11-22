package providers

import (
	"bytes"
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"html/template"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/flying-robot/commute-agent/internal"
	"github.com/labstack/echo/v5"
)

// conversationCookie is the cookie that has the conversation ID of a browser, so
// that each browser has its own conversation with the agent.
const conversationCookie = "conversation"

// page is the chat page of the agent. It uses HTMX to send each message with
// POST /messages, and shows the progress and the reply of the agent from an
// event stream.
//
//go:embed http_provider.html
var page string

// placeholderTemplate is the HTML of a reply while the agent works. It opens the
// event stream of the turn, adds each step event to its progress list, and is
// replaced by the done event. It closes the stream at the done event, so that
// the browser does not connect again.
var placeholderTemplate = template.Must(template.New("placeholder").Parse(`
<div class="msg assistant turn" hx-ext="sse" sse-connect="/messages/{{.}}/events" sse-close="done">
  <div class="progress" sse-swap="step" hx-swap="beforeend"></div>
  <div class="typing" aria-label="The agent is working"><span></span><span></span><span></span></div>
  <div sse-swap="done" hx-target="closest .turn" hx-swap="outerHTML"></div>
</div>`))

// progressTemplate is the HTML of one step event: a reply of the model, with
// the tools that it calls, or the result of one tool.
var progressTemplate = template.Must(template.New("progress").Parse(`
{{if .Tool}}<div class="progress-line {{if .Failed}}failed{{else}}ok{{end}}">{{.Tool}} {{if .Failed}}failed{{else}}done{{end}} <span class="time">{{.Duration}}</span></div>
{{else}}<div class="progress-line">Step {{.Step}} <span class="time">model: {{.Duration}}</span>{{if .Calls}}, calling {{.Calls}}{{else}}, writing the answer{{end}}</div>{{end}}`))

// replyTemplate is the HTML of one reply on the chat page. The page renders the
// Markdown of the answer, so the template only escapes it. The steps are in a
// closed details element, so that they show only when the user opens them.
var replyTemplate = template.Must(template.New("reply").Parse(`
<div class="msg {{if .Failed}}error{{else}}assistant{{end}}">
  {{if .Failed}}<div>{{.Answer}}</div>{{else}}<div data-md>{{.Answer}}</div>{{end}}
  {{with .Notice}}<div class="notice">{{.}}</div>{{end}}
  {{if .Calls}}<div class="pills">{{range .Calls}}<span class="pill {{if .Failed}}failed{{else}}ok{{end}}" title="{{if .Failed}}{{.Result}}{{else}}Succeeded{{end}}">{{.Name}} <span class="time">{{.Duration}}</span></span>{{end}}</div>{{end}}
  {{if .Steps}}<details class="trace">
    <summary>How the agent got this answer (steps: {{len .Steps}}, tool calls: {{len .Calls}}, total time: {{.Total}})</summary>
    {{range $step := .Steps}}<div class="step">
      <div class="step-title">Step {{$step.Number}} <span class="time">model: {{$step.Duration}}</span></div>
      {{with $step.Thinking}}<div class="label">Thinking</div><pre>{{.}}</pre>{{end}}
      {{range $step.Calls}}<div class="label">Tool call: {{.Name}} <span class="time">{{.Duration}}</span></div><pre>{{.Arguments}}</pre>
      <div class="label {{if .Failed}}failed{{end}}">{{if .Failed}}Failed{{else}}Result{{end}}</div><pre>{{.Result}}</pre>{{end}}
      {{with $step.Content}}<div class="label">Answer text</div><pre>{{.}}</pre>{{end}}
    </div>{{end}}
  </details>{{end}}
</div>`))

// alertsTemplate is the HTML of the requests for feedback at the top of the chat
// page. A pending request has fields for the answer. The alert after an answer
// replaces the alert, with the error of the answer if there is one.
var alertsTemplate = template.Must(template.New("alerts").Funcs(template.FuncMap{"ratings": func() []int { return []int{5, 4, 3, 2, 1} }}).Parse(`{{range .}}{{template "alert" .}}{{end}}
{{define "alert"}}<div class="alert {{.Status}}" id="alert-{{.ID}}">
  <div>{{.Text}}</div>
  {{if eq .Status "pending"}}<div class="alert-actions feedback">
    <input type="number" name="actual_minutes" min="1" placeholder="Minutes" aria-label="Actual travel time in minutes">
    <select name="rating" aria-label="Rating">{{range $r := ratings}}<option value="{{$r}}">{{$r}}</option>{{end}}</select>
    <input type="text" name="comment" placeholder="What went wrong? (needed for a rating of 1 or 2)" aria-label="Comment">
    <button type="button" hx-post="/alerts/{{.ID}}/feedback" hx-include="closest .feedback" hx-target="#alert-{{.ID}}" hx-swap="outerHTML">Send</button>
  </div>{{end}}
  {{with .Error}}<div class="alert-error">{{.}}</div>{{end}}
</div>{{end}}`))

// alertView is the data of one alert for alertsTemplate.
type alertView struct {
	internal.Alert
	Error string
}

// feedbackRequest is the body of POST /alerts/{id}/feedback, as JSON or as a
// form.
type feedbackRequest struct {
	ActualMinutes int    `json:"actual_minutes" form:"actual_minutes"`
	Rating        int    `json:"rating" form:"rating"`
	Comment       string `json:"comment" form:"comment"`
}

// pendingTurn is a message from the chat page that waits for its event stream.
type pendingTurn struct {
	conversation string
	content      string
}

// progressView is the data of one step event for progressTemplate.
type progressView struct {
	Step     int
	Duration string
	Calls    string
	Tool     string
	Failed   bool
}

// replyView is the data of one reply for replyTemplate.
type replyView struct {
	Answer string
	Notice string
	Failed bool
	Total  string
	Steps  []stepView
	Calls  []*callView
}

// stepView is one reply of the model in a turn, with its tool calls.
type stepView struct {
	Number   int
	Duration string
	Thinking string
	Content  string
	Calls    []*callView
}

// callView is one tool call and its result.
type callView struct {
	Name      string
	Arguments string
	Result    string
	Failed    bool
	Duration  string
}

// messageRequest is the body of POST /messages, as JSON or as a form.
type messageRequest struct {
	Content string `json:"content" form:"content"`
}

// messageResponse is the reply of the agent to one message. Notice says that the
// answer used saved data, if it did.
type messageResponse struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Notice  string `json:"notice,omitempty"`
}

// NewHTTPProvider returns an internal.ServeFunc that exposes the agent as a
// RESTful API on addr. GET / serves the chat page, and GET /maps/ serves the
// route maps from the maps handler. The server stops gracefully when the
// context is canceled.
//
// A caller sends a message with POST /messages. A request from the chat page
// gets a placeholder at once, which opens GET /messages/{id}/events. That event
// stream runs the turn, sends each step of the agent as it happens, and sends
// the reply at the end. Other callers wait for the answer as JSON.
//
// Each browser has a conversation ID in a cookie, and POST /conversations
// starts a new conversation.
//
// GET /alerts gives the requests for feedback of the agent, and POST
// /alerts/{id}/feedback answers one. The question whether to delegate the
// pickup is in the chat. The chat page shows the alerts as HTML, and
// other callers get JSON.
func NewHTTPProvider(addr string, maps http.Handler) internal.ServeFunc {
	return func(ctx context.Context, agent internal.Agent) error {
		respond := agent.Respond
		e := echo.New()

		// pending has the messages from the chat page that wait for their event
		// stream, by the ID of the turn. The turn runs only when the stream opens,
		// so that no step of the agent is lost before the page listens.
		var mu sync.Mutex
		pending := map[string]pendingTurn{}

		e.GET("/", func(c *echo.Context) error {
			return c.HTML(http.StatusOK, page)
		})

		e.GET("/maps/*", echo.WrapHandler(http.StripPrefix("/maps/", maps)))

		// A new conversation ID starts a new conversation. The page reloads
		// itself after this request, so that it shows no old messages.
		e.POST("/conversations", func(c *echo.Context) error {
			newConversation(c)
			return c.NoContent(http.StatusNoContent)
		})

		e.POST("/messages", func(c *echo.Context) error {
			// Decode the request body, which is JSON or a form from the chat page,
			// and return 400 if it is not valid.
			var req messageRequest
			if err := c.Bind(&req); err != nil {
				return replyError(c, http.StatusBadRequest, "the request body is not valid")
			}

			// Reject an empty message, so that it does not reach the model or stay
			// in the conversation.
			if strings.TrimSpace(req.Content) == "" {
				return replyError(c, http.StatusBadRequest, "content is required")
			}
			conversation := conversationID(c)

			// Give the chat page a placeholder at once. It opens the event stream
			// of the turn, which runs the turn.
			if isHTMX(c) {
				id := rand.Text()
				mu.Lock()
				pending[id] = pendingTurn{conversation: conversation, content: req.Content}
				mu.Unlock()
				return c.HTML(http.StatusOK, render(placeholderTemplate, id))
			}

			// Other callers wait for the answer as JSON. The request context is
			// canceled if the caller disconnects, which also stops the agent.
			turn, err := respond(c.Request().Context(), conversation, req.Content, func(context.Context, internal.Message) {})
			if err != nil {
				return echo.NewHTTPError(http.StatusBadGateway, err.Error())
			}
			return c.JSON(http.StatusOK, messageResponse{
				Role:    "assistant",
				Content: withMentions(turn, turn[len(turn)-1].Content),
				Notice:  savedNotice(turn),
			})
		})

		e.GET("/messages/:id/events", func(c *echo.Context) error {
			// First, take the message of the turn. An unknown ID, for example when
			// the browser connects again after the turn has ended, gets no content,
			// which makes the browser stop connecting.
			id := c.Param("id")
			mu.Lock()
			p, ok := pending[id]
			delete(pending, id)
			mu.Unlock()
			if !ok {
				return c.NoContent(http.StatusNoContent)
			}

			// Next, run the turn, and send each reply of the model and each tool
			// result as a step event as soon as the agent has it. The request
			// context is canceled if the page closes, which also stops the agent.
			w := c.Response()
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.WriteHeader(http.StatusOK)
			step := 0
			turn, err := respond(c.Request().Context(), p.conversation, p.content, func(_ context.Context, m internal.Message) {
				if m.Role == "assistant" {
					step++
				}
				writeEvent(w, "step", render(progressTemplate, newProgressView(step, m)))
			})

			// Finally, send the reply with all its steps as the done event, which
			// replaces the placeholder. A failed turn shows its steps too.
			answer := ""
			if err != nil {
				answer = err.Error()
			} else {
				answer = withMentions(turn, turn[len(turn)-1].Content)
			}
			writeEvent(w, "done", render(replyTemplate, newReplyView(turn, answer, err != nil)))
			return nil
		})

		e.GET("/alerts", func(c *echo.Context) error {
			alerts := agent.Alerts(c.Request().Context())
			if !isHTMX(c) {
				return c.JSON(http.StatusOK, alerts)
			}
			views := []alertView{}
			for _, a := range alerts {
				views = append(views, alertView{Alert: a})
			}
			return c.HTML(http.StatusOK, render(alertsTemplate, views))
		})

		e.POST("/alerts/:id/feedback", func(c *echo.Context) error {
			var req feedbackRequest
			if err := c.Bind(&req); err != nil {
				return replyAlert(c, agent, internal.Alert{}, errors.New("give the actual travel time as a number of minutes"))
			}
			alert, err := agent.Feedback(c.Request().Context(), c.Param("id"), req.ActualMinutes, req.Rating, req.Comment)
			return replyAlert(c, agent, alert, err)
		})

		// Start the HTTP server with the given configuration.
		return echo.StartConfig{
			Address: addr,
		}.Start(ctx, e)
	}
}

// conversationID returns the conversation ID of the browser, and starts a new
// conversation if the browser has none. A caller without cookies, such as curl
// without a cookie file, gets a new conversation for each message.
func conversationID(c *echo.Context) string {
	if cookie, err := c.Cookie(conversationCookie); err == nil && cookie.Value != "" {
		return cookie.Value
	}
	return newConversation(c)
}

// newConversation gives the browser a new random conversation ID. The cookie is
// not readable from JavaScript, because the page does not need it.
func newConversation(c *echo.Context) string {
	id := rand.Text()
	c.SetCookie(&http.Cookie{
		Name:     conversationCookie,
		Value:    id,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
	})
	return id
}

// writeEvent sends one server-sent event and flushes it, so that the browser
// gets it at once. Each line of the data needs its own data field.
func writeEvent(w http.ResponseWriter, event, data string) {
	fmt.Fprintf(w, "event: %s\n", event)
	for line := range strings.SplitSeq(data, "\n") {
		fmt.Fprintf(w, "data: %s\n", line)
	}
	fmt.Fprint(w, "\n")
	http.NewResponseController(w).Flush()
}

// render returns the HTML of a template with its data. The templates are fixed
// and their data has the expected fields, so an error is only possible after a
// change to a template, and the error text then shows on the page.
func render(t *template.Template, data any) string {
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		return html.EscapeString(err.Error())
	}
	return buf.String()
}

// isHTMX reports whether a request comes from HTMX on the chat page.
func isHTMX(c *echo.Context) bool {
	return c.Request().Header.Get("HX-Request") == "true"
}

// replyError returns an error as an HTML message for the chat page, and as a
// JSON error for other callers.
func replyError(c *echo.Context, status int, msg string) error {
	if isHTMX(c) {
		return c.HTML(status, `<div class="msg error">`+html.EscapeString(msg)+`</div>`)
	}
	return echo.NewHTTPError(status, msg)
}

// replyAlert returns the alert after an answer: as HTML for the chat page, with
// the error under the alert, and as JSON for other callers. After an error, the
// chat page gets the alert as it is now, so that the user can answer again.
func replyAlert(c *echo.Context, agent internal.Agent, alert internal.Alert, err error) error {
	if !isHTMX(c) {
		if err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, err.Error())
		}
		return c.JSON(http.StatusOK, alert)
	}
	view := alertView{Alert: alert}
	if err != nil {
		view.Error = err.Error()
		for _, a := range agent.Alerts(c.Request().Context()) {
			if a.ID == c.Param("id") {
				view.Alert = a
			}
		}
		view.ID = c.Param("id")
	}
	return c.HTML(http.StatusOK, render(alertsTemplate.Lookup("alert"), view))
}

// newProgressView makes the data of one step event from a message of a turn.
func newProgressView(step int, m internal.Message) progressView {
	view := progressView{Step: step, Duration: seconds(m.Duration)}
	if m.Role == "tool" {
		view.Tool = m.ToolName
		view.Failed = strings.HasPrefix(m.Content, internal.ToolFailed)
		return view
	}
	var names []string
	for _, c := range m.ToolCalls {
		names = append(names, c.Name)
	}
	view.Calls = strings.Join(names, ", ")
	return view
}

// newReplyView makes the data of one reply from the messages of a turn. Each
// reply of the model starts a step, and the tool results that follow it belong
// to its tool calls in the same order. The total time is the time of all the
// model calls and tool calls together.
func newReplyView(turn []internal.Message, answer string, failed bool) replyView {
	view := replyView{Answer: answer, Notice: savedNotice(turn), Failed: failed}
	next := 0
	var total time.Duration
	for _, m := range turn {
		total += m.Duration
		switch m.Role {
		case "assistant":
			step := stepView{Number: len(view.Steps) + 1, Duration: seconds(m.Duration), Thinking: m.Thinking, Content: m.Content}
			for _, c := range m.ToolCalls {
				call := &callView{Name: c.Name, Arguments: indent(string(c.Arguments))}
				step.Calls = append(step.Calls, call)
				view.Calls = append(view.Calls, call)
			}
			view.Steps = append(view.Steps, step)
			next = len(view.Calls) - len(step.Calls)
		case "tool":
			if next < len(view.Calls) {
				view.Calls[next].Result = indent(m.Content)
				view.Calls[next].Failed = strings.HasPrefix(m.Content, internal.ToolFailed)
				view.Calls[next].Duration = seconds(m.Duration)
				next++
			}
		}
	}
	view.Total = seconds(total)
	return view
}

// savedNotice returns a notice for the user if a tool result of the turn has
// saved data, with the age of the oldest saved data, and an empty string if
// not. The model does not reliably tell the user that it did not check again,
// so the provider does.
func savedNotice(turn []internal.Message) string {
	// First, find the age of the oldest saved data in the JSON tool results.
	oldest := -1
	for _, result := range toolResults(turn) {
		if age, ok := result[internal.SavedMinutesAgo].(float64); ok {
			oldest = max(oldest, int(age))
		}
	}

	// Finally, write the notice.
	when := fmt.Sprintf("%d minutes ago", oldest)
	switch oldest {
	case -1:
		return ""
	case 0:
		when = "less than a minute ago"
	case 1:
		when = "1 minute ago"
	}
	return "I did not check some data again, because I checked it " + when + ". Ask me to check again to get new data."
}

// withMentions returns the answer with the mention text of each tool result of
// the turn added at the end, if the answer does not have it yet, so that the
// user always sees it.
func withMentions(turn []internal.Message, answer string) string {
	for _, result := range toolResults(turn) {
		if text, _ := result[internal.Mention].(string); text != "" && !strings.Contains(answer, text) {
			answer += "\n\n" + text
		}
	}
	return answer
}

// toolResults returns the tool results of a turn that are JSON objects. A
// failure is text, so it is not in the list.
func toolResults(turn []internal.Message) []map[string]any {
	var results []map[string]any
	for _, m := range turn {
		var result map[string]any
		if m.Role == "tool" && json.Unmarshal([]byte(m.Content), &result) == nil {
			results = append(results, result)
		}
	}
	return results
}

// seconds returns a duration in seconds with one decimal place, for example
// "2.3 s", which is exact enough to find the slow steps.
func seconds(d time.Duration) string {
	return fmt.Sprintf("%.1f s", d.Seconds())
}

// indent returns JSON with indentation, so that it is easy to read, and other
// text without a change.
func indent(s string) string {
	var buf bytes.Buffer
	if json.Indent(&buf, []byte(s), "", "  ") != nil {
		return s
	}
	return buf.String()
}
