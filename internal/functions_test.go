package internal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

// tools are given to the agent in every test. Together they cover each outcome
// of a tool call: echo succeeds, fail returns an error, and any other name is unknown.
var tools = []Tool{
	{Name: "echo", Run: func(_ context.Context, args json.RawMessage) (string, error) {
		return string(args), nil
	}},
	{Name: "fail", Run: func(context.Context, json.RawMessage) (string, error) {
		return "", errors.New("broken")
	}},
}

// toolCalls is a model reply that asks for one call of each kind of tool.
var toolCalls = Message{Role: "assistant", ToolCalls: []ToolCall{
	{Name: "echo", Arguments: json.RawMessage(`1`)},
	{Name: "fail"},
	{Name: "missing"},
}}

// toolResults are the observations that the agent adds for toolCalls.
var toolResults = []Message{
	{Role: "tool", ToolName: "echo", Content: "1"},
	{Role: "tool", ToolName: "fail", Content: ToolFailed + "broken"},
	{Role: "tool", ToolName: "missing", Content: ToolFailed + "unknown tool missing"},
}

// A reply is one scripted result that the fake model returns.
type reply struct {
	msg Message
	err error
}

// TestRespond sends each message in turns to one agent, and checks the
// result of the last turn and the messages of the last call to the model. The
// earlier turns let a test check what stays in the conversation.
func TestRespond(t *testing.T) {
	system := Message{Role: "system", Content: plannerPrompt}
	hi := Message{Role: "user", Content: "hi"}
	again := Message{Role: "user", Content: "again"}
	hello := Message{Role: "assistant", Content: "hello"}

	tests := []struct {
		name     string
		turns    []string
		replies  []reply
		want     string
		wantErr  string
		wantLast []Message
		wantTurn []Message
	}{
		{
			name:     "direct answer",
			turns:    []string{"hi"},
			replies:  []reply{{msg: hello}},
			want:     "hello",
			wantLast: []Message{system, hi},
		},
		{
			name:     "tool results are observations",
			turns:    []string{"hi"},
			replies:  []reply{{msg: toolCalls}, {msg: hello}},
			want:     "hello",
			wantLast: append([]Message{system, hi, toolCalls}, toolResults...),
			wantTurn: append(append([]Message{hi, toolCalls}, toolResults...), hello),
		},
		{
			name:     "chat error",
			turns:    []string{"hi"},
			replies:  []reply{{err: errors.New("model down")}},
			wantErr:  "step 1: model down",
			wantLast: []Message{system, hi},
			wantTurn: []Message{hi},
		},
		{
			name:    "no answer within the step limit",
			turns:   []string{"hi"},
			replies: slices.Repeat([]reply{{msg: toolCalls}}, maxSteps),
			wantErr: fmt.Sprintf("no answer after %d steps", maxSteps),
		},
		{
			name:     "answered turn is kept",
			turns:    []string{"hi", "again"},
			replies:  []reply{{msg: hello}, {msg: hello}},
			want:     "hello",
			wantLast: []Message{system, hi, hello, again},
			wantTurn: []Message{again, hello},
		},
		{
			name:     "failed turn is dropped",
			turns:    []string{"hi", "again"},
			replies:  []reply{{msg: toolCalls}, {err: errors.New("model down")}, {msg: hello}},
			want:     "hello",
			wantLast: []Message{system, again},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// First, create a fake model that returns the replies in order, and keeps
			// a copy of the messages that it gets in each call.
			var calls [][]Message
			chat := func(_ context.Context, messages []Message, _ []Tool, _ bool) (Message, error) {
				if len(calls) == len(tt.replies) {
					t.Fatalf("unexpected chat call %d", len(calls)+1)
				}
				calls = append(calls, slices.Clone(messages))
				r := tt.replies[len(calls)-1]
				return r.msg, r.err
			}
			// The clock never moves, so that each duration is zero and the expected
			// messages do not need durations.
			log := func(context.Context, Level, string, ...any) {}
			now := func(context.Context) time.Time { return time.Time{} }
			agent := NewAgent(Actions{Chat: chat, Log: log, Now: now, Planner: tools}).Respond

			// Next, send each turn to the agent in one conversation, and keep the
			// result of the last one, and the messages that the agent gave while it
			// worked on it. The answer is the last message of a turn that did not
			// fail.
			var turn, streamed []Message
			var err error
			for _, content := range tt.turns {
				streamed = nil
				turn, err = agent(t.Context(), "conversation", content, func(_ context.Context, m Message) {
					streamed = append(streamed, m)
				})
			}
			got := ""
			if err == nil {
				got = turn[len(turn)-1].Content
			}

			// Next, check the answer or the error of the last turn.
			gotErr := ""
			if err != nil {
				gotErr = err.Error()
			}
			if gotErr != tt.wantErr {
				t.Errorf("error = %q, want %q", gotErr, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("answer = %q, want %q", got, tt.want)
			}
			if tt.wantTurn != nil && !reflect.DeepEqual(turn, tt.wantTurn) {
				t.Errorf("turn messages =\n%v\nwant\n%v", turn, tt.wantTurn)
			}
			// Each message of the turn after the message of the caller must also
			// reach the MessageFunc, in the same order.
			if !slices.EqualFunc(streamed, turn[1:], func(a, b Message) bool { return reflect.DeepEqual(a, b) }) {
				t.Errorf("streamed messages =\n%v\nwant\n%v", streamed, turn[1:])
			}

			// Finally, check that the model got each reply, and that the last call
			// had the expected conversation.
			if len(calls) != len(tt.replies) {
				t.Errorf("chat calls = %d, want %d", len(calls), len(tt.replies))
			}
			if last := calls[len(calls)-1]; tt.wantLast != nil && !reflect.DeepEqual(last, tt.wantLast) {
				t.Errorf("last chat messages =\n%v\nwant\n%v", last, tt.wantLast)
			}
		})
	}
}

// TestConversationsAreSeparate checks that a turn in one conversation does not
// add to the history of a different conversation.
func TestConversationsAreSeparate(t *testing.T) {
	var last []Message
	chat := func(_ context.Context, messages []Message, _ []Tool, _ bool) (Message, error) {
		last = slices.Clone(messages)
		return Message{Role: "assistant", Content: "hello"}, nil
	}
	log := func(context.Context, Level, string, ...any) {}
	now := func(context.Context) time.Time { return time.Time{} }
	ignore := func(context.Context, Message) {}
	agent := NewAgent(Actions{Chat: chat, Log: log, Now: now, Planner: tools}).Respond

	agent(t.Context(), "a", "hi", ignore)
	agent(t.Context(), "b", "again", ignore)

	want := []Message{{Role: "system", Content: plannerPrompt}, {Role: "user", Content: "again"}}
	if !reflect.DeepEqual(last, want) {
		t.Errorf("messages of conversation b =\n%v\nwant\n%v", last, want)
	}
}

// TestDecide checks the level of the decider at each boundary of its
// thresholds, and that only the levels from reroute up ask the contact.
func TestDecide(t *testing.T) {
	route := func(delay float64) map[string]any {
		return map[string]any{"traffic_delay_minutes": delay, "travel_minutes": 50.0}
	}
	tests := []struct {
		name      string
		route     map[string]any
		alert     bool
		extra     int
		wantLevel int
		wantAsk   bool
		force     bool
	}{
		{"no delay", route(0), false, 0, onTime, false, false},
		{"minor delay at the limit", route(15), false, 0, leaveEarlier, false, false},
		{"major delay above the minor limit", route(16), false, 0, reroute, true, false},
		{"major delay at the limit", route(30), false, 0, reroute, true, false},
		{"extreme delay", route(31), false, 0, delegate, true, false},
		{"weather alert without delay", route(0), true, 0, reroute, true, false},
		{"feedback adds minutes", route(10), false, 10, reroute, true, false},
		{"no route", nil, false, 0, noRoute, true, false},
		{"forced without delay", route(0), false, 0, delegate, true, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := decide(tt.route, tt.alert, tt.extra, "Bob", 15, 30, tt.force)
			if d.level != tt.wantLevel {
				t.Errorf("level = %d, want %d", d.level, tt.wantLevel)
			}
			if asks := strings.Contains(d.text, "ask Bob"); asks != tt.wantAsk {
				t.Errorf("text = %q, asks Bob = %v, want %v", d.text, asks, tt.wantAsk)
			}
		})
	}
}

// TestDelegate checks that an accepted offer sends one pickup request, that the
// request is sent at most once on a day, and that a failure is retried, except
// a failure with no response.
func TestDelegate(t *testing.T) {
	plan := pickupPlan{name: "Alice", destination: "Daycare", closes: "18:00", contact: "Bob"}
	now := time.Date(2026, 11, 16, 15, 0, 0, 0, time.UTC)
	tests := []struct {
		name      string
		failures  []error
		wantSends int
		wantSent  bool
	}{
		{"sent", nil, 1, true},
		{"retried after errors", []error{errors.New("status 500"), errors.New("status 429")}, 3, true},
		{"not retried with no response", []error{ErrNoResponse}, 1, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sends := 0
			a := Actions{
				Log:   func(context.Context, Level, string, ...any) {},
				Now:   func(context.Context) time.Time { return now },
				Sleep: func(context.Context, time.Duration) error { return nil },
				Notify: func(_ context.Context, _, _, body string) (string, error) {
					sends++
					if !strings.Contains(body, "Alice asks you to do the pickup at Daycare today, before 18:00") {
						t.Errorf("body = %q, want the template", body)
					}
					if sends <= len(tt.failures) {
						return "", tt.failures[sends-1]
					}
					return "id", nil
				},
			}
			s := &pickupState{sent: map[string]bool{}}
			s.offer(now, decide(map[string]any{"traffic_delay_minutes": 40.0}, false, 0, "Bob", 15, 30, false), plan)
			s.offer(now, decide(map[string]any{"traffic_delay_minutes": 45.0}, false, 0, "Bob", 15, 30, false), plan)
			if n := len(s.alerts); n != 1 {
				t.Fatalf("alerts = %d, want one offer for the day", n)
			}

			alert, err := s.delegate(t.Context(), a, s.alerts[0].ID, true)
			if err != nil {
				t.Fatal(err)
			}
			if sends != tt.wantSends {
				t.Errorf("sends = %d, want %d", sends, tt.wantSends)
			}
			if sent := alert.Status == "sent"; sent != tt.wantSent {
				t.Errorf("status = %q, want sent %v", alert.Status, tt.wantSent)
			}

			// A second answer to the same offer must not send again, and a sent
			// request means no new offer on that day.
			if _, err := s.delegate(t.Context(), a, s.alerts[0].ID, true); err == nil {
				t.Error("second answer: no error, want one")
			}
			text := s.offer(now, decide(map[string]any{"traffic_delay_minutes": 50.0}, false, 0, "Bob", 15, 30, false), plan)
			if tt.wantSent && (len(s.alerts) != 1 || !strings.Contains(text, "already sent")) {
				t.Errorf("alerts = %d, text = %q, want no new offer after a sent request", len(s.alerts), text)
			}
		})
	}
}

// TestCheckTimes checks that the pickup checks run 90 minutes before the
// departure, and every 5 minutes in the last 30 minutes, only on the days of the
// plan.
func TestCheckTimes(t *testing.T) {
	plan := parsePlan("leave_by: 16:45\ndays: Monday, Friday")
	monday := time.Date(2026, 11, 16, 9, 0, 0, 0, time.UTC)
	var got []string
	for _, c := range checkTimes(plan, monday) {
		got = append(got, c.Format("15:04"))
	}
	want := []string{"15:15", "16:15", "16:20", "16:25", "16:30", "16:35", "16:40", "16:45"}
	if !slices.Equal(got, want) {
		t.Errorf("times = %v, want %v", got, want)
	}
	if times := checkTimes(plan, monday.AddDate(0, 0, 1)); times != nil {
		t.Errorf("times on Tuesday = %v, want none", times)
	}
}

// TestYesOrNo checks which replies answer the question whether to delegate the
// pickup. Only a clear first word counts, so that an unclear reply sends nothing.
func TestYesOrNo(t *testing.T) {
	tests := []struct {
		reply      string
		wantAccept bool
		wantOK     bool
	}{
		{"yes", true, true},
		{"Yes, please ask him.", true, true},
		{"OK", true, true},
		{"no", false, true},
		{"No thanks", false, true},
		{"Don't send it", false, true},
		{"not now", false, true},
		{"What about the weather?", false, false},
		{"maybe", false, false},
		{"", false, false},
	}
	for _, tt := range tests {
		accept, ok := yesOrNo(tt.reply)
		if accept != tt.wantAccept || ok != tt.wantOK {
			t.Errorf("yesOrNo(%q) = %v, %v, want %v, %v", tt.reply, accept, ok, tt.wantAccept, tt.wantOK)
		}
	}
}

// world is a fake world for full turns: a trip planner and a small model that
// follow a script, and fake tools, stores, notifications, and a clock. The
// clock moves only when the agent sleeps, and stops the agent at stop.
type world struct {
	plan, route, weather, news string
	trips                      []Document
	saved                      []Document
	sent                       []string
	plannerCalls               [][]Message
	now, stop                  time.Time
}

// newWorld returns a world with a pickup plan for Alice with Bob as the
// contact, and a route with a delay in minutes, no weather alert, and one news
// article.
func newWorld(delay int) *world {
	return &world{
		plan:    "name: Alice\nstart: Waldorf, MD\ndestination: Alexandria, VA\nleave_by: 10:00\ncloses: 18:00\ndays: Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday\ncontact: Bob",
		route:   testRoute(delay),
		weather: `{"start":{"weather":{"alerts":[]}},"destination":{"weather":{"alerts":[]}}}`,
		news:    `{"articles":[],"mention":"**News:** WTOP has no articles related to this trip."}`,
		now:     time.Date(2026, 11, 16, 8, 0, 0, 0, time.UTC),
	}
}

// testRoute returns a route result with a delay in minutes, one other route, a
// map, and two steps.
func testRoute(delay int) string {
	b, _ := json.Marshal(map[string]any{
		"start": "Waldorf, MD", "destination": "Alexandria, VA", "departure": "2026-11-16T10:00:00Z", "arrival": "2026-11-16T10:50:00Z",
		"travel_minutes": 50, "traffic_delay_minutes": delay, "news_about": "Waldorf, MD",
		"other_routes": []map[string]any{{"option": 2, "via": []string{"US-301", "MD-223"}, "travel_minutes": 55}},
		"map":          "/maps/test.png",
		"steps":        []string{"Head north", "Arrive"},
	})
	return string(b)
}

// result returns a tool that gives out, or fails if out is empty.
func result(name string, out *string) Tool {
	return Tool{Name: name, Run: func(context.Context, json.RawMessage) (string, error) {
		if *out == "" {
			return "", errors.New("service unavailable")
		}
		return *out, nil
	}}
}

// actions returns the actions of the world. The trip planner asks for the route
// once after each message that asks about a drive, and then answers. The small
// model calls both conditions tools as the conditions agent, and writes
// "report" as the reporter.
func (w *world) actions() Actions {
	return Actions{
		Chat: func(_ context.Context, messages []Message, _ []Tool, _ bool) (Message, error) {
			w.plannerCalls = append(w.plannerCalls, slices.Clone(messages))
			i := len(messages) - 1
			for i >= 0 && messages[i].Role != "user" {
				i--
			}
			asked := slices.ContainsFunc(messages[i:], func(m Message) bool { return m.Role == "tool" && m.ToolName == routeTool })
			if strings.Contains(messages[i].Content, "drive") && !asked {
				return Message{Role: "assistant", ToolCalls: []ToolCall{{Name: routeTool, Arguments: json.RawMessage(`{}`)}}}, nil
			}
			return Message{Role: "assistant", Content: "planner answer"}, nil
		},
		SmallChat: func(_ context.Context, messages []Message, _ []Tool, _ bool) (Message, error) {
			if messages[0].Content == conditionsPrompt {
				return Message{Role: "assistant", ToolCalls: []ToolCall{{Name: "get_route_weather"}, {Name: "get_current_news"}}}, nil
			}
			return Message{Role: "assistant", Content: "report"}, nil
		},
		Log: func(context.Context, Level, string, ...any) {},
		Now: func(context.Context) time.Time { return w.now },
		Sleep: func(ctx context.Context, d time.Duration) error {
			w.now = w.now.Add(d)
			if !w.stop.IsZero() && w.now.After(w.stop) {
				return context.Canceled
			}
			return ctx.Err()
		},
		Planner: []Tool{
			{Name: memoryTool, Run: func(context.Context, json.RawMessage) (string, error) {
				if w.plan == "" {
					return `{"memories":[]}`, nil
				}
				b, _ := json.Marshal(map[string]any{"memories": []map[string]string{{"name": PickupFact, "fact": w.plan}}})
				return string(b), nil
			}},
			{Name: "get_current_time", Run: func(context.Context, json.RawMessage) (string, error) { return "{}", nil }},
			result(routeTool, &w.route),
			result("search_saved_data", &w.news),
			{Name: "save_memory", Run: func(context.Context, json.RawMessage) (string, error) { return "saved", nil }},
		},
		Conditions: []Tool{result("get_route_weather", &w.weather), result("get_current_news", &w.news)},
		Notify: func(_ context.Context, to, title, body string) (string, error) {
			w.sent = append(w.sent, to+": "+title+": "+body)
			return "id", nil
		},
		SaveTrip: func(_ context.Context, _ string, docs []Document, _ time.Duration) error {
			w.saved = append(w.saved, docs...)
			return nil
		},
		Trips: func(context.Context, string, string, string, int) ([]Document, error) {
			return w.trips, nil
		},
		MinorDelay: 15,
		MajorDelay: 30,
	}
}

// ask sends one message to the agent in conversation "c", and returns the answer.
func ask(t *testing.T, ag Agent, content string) string {
	t.Helper()
	turn, err := ag.Respond(t.Context(), "c", content, func(context.Context, Message) {})
	if err != nil {
		t.Fatalf("%q: %v", content, err)
	}
	return turn[len(turn)-1].Content
}

// TestTurnWithRoute checks a full turn with a route: the recommendation is
// first, then the report, the news, the map, and the steps, and the question
// whether to delegate the pickup is last. An unclear reply gets the question
// again, a yes sends one request, and after that no question is asked. The
// history of the trip planner never has the question.
func TestTurnWithRoute(t *testing.T) {
	w := newWorld(45)
	ag := NewAgent(w.actions())

	answer := ask(t, ag, "How is the drive?")
	for _, want := range []string{"**Recommendation:** The delay is 45 minutes, so ask Bob to do the pickup.", "report", "**News:** WTOP has no articles", "![Route map](/maps/test.png)", "1. Head north\n2. Arrive"} {
		if !strings.Contains(answer, want) {
			t.Errorf("answer does not have %q:\n%s", want, answer)
		}
	}
	if !strings.HasSuffix(answer, "**Do you want me to ask Bob to do the pickup today, because the delay is 45 minutes?** Reply yes or no.") {
		t.Errorf("answer does not end with the question:\n%s", answer)
	}

	if answer := ask(t, ag, "What about the weather?"); !strings.HasPrefix(answer, "planner answer") || !strings.HasSuffix(answer, "Reply yes or no.") {
		t.Errorf("unclear reply: answer = %q, want the planner answer and the question again", answer)
	}
	if answer := ask(t, ag, "Yes, please."); !strings.HasPrefix(answer, "The pickup request was sent to Bob.") || !strings.Contains(answer, "**Message ID:** id") {
		t.Errorf("yes: answer = %q", answer)
	}
	if len(w.sent) != 1 || w.sent[0] != "Bob: Pickup request: Pickup request from the commute agent: Alice asks you to do the pickup at Alexandria, VA today, before 18:00, because the delay is 45 minutes." {
		t.Errorf("sent = %q, want one pickup request", w.sent)
	}
	if answer := ask(t, ag, "yes"); answer != "planner answer" {
		t.Errorf("second yes: answer = %q, want the planner answer with no question", answer)
	}
	if answer := ask(t, ag, "How is the drive now?"); !strings.Contains(answer, "already sent to Bob today") || strings.Contains(answer, "Reply yes or no.") {
		t.Errorf("after the request: answer = %q, want no new offer", answer)
	}
	for _, call := range w.plannerCalls {
		for _, m := range call {
			if strings.Contains(m.Content, "Do you want me") {
				t.Fatalf("the trip planner saw the question: %q", m.Content)
			}
		}
	}
}

// TestTurnReject checks that a no sends nothing, and that a plan without a
// contact makes no offer.
func TestTurnReject(t *testing.T) {
	w := newWorld(20)
	ag := NewAgent(w.actions())
	ask(t, ag, "How is the drive?")
	if answer := ask(t, ag, "No thanks"); answer != "OK. I will not ask Bob to do the pickup today." || len(w.sent) != 0 {
		t.Errorf("no: answer = %q, sent = %q", answer, w.sent)
	}

	w = newWorld(45)
	w.plan = "name: Alice"
	if answer := ask(t, NewAgent(w.actions()), "How is the drive?"); strings.Contains(answer, "Reply yes or no.") || !strings.Contains(answer, "one of the other routes: option 2 via US-301, MD-223 (55 minutes)") {
		t.Errorf("no contact: answer = %q, want other routes and no question", answer)
	}
}

// TestTurnWeatherAndFeedback checks that a weather alert offers other routes,
// and that the feedback of earlier trips adds the average extra minutes.
func TestTurnWeatherAndFeedback(t *testing.T) {
	w := newWorld(0)
	w.weather = `{"start":{"weather":{"alerts":["Flood Warning"]}},"destination":{"weather":null},"saved_minutes_ago":3}`
	if answer := ask(t, NewAgent(w.actions()), "How is the drive?"); !strings.Contains(answer, "There is a weather alert on the route. Take one of the other routes") {
		t.Errorf("weather alert: answer = %q", answer)
	}

	w = newWorld(0)
	w.trips = []Document{
		{ID: "2026-11-13", Text: "estimated_minutes: 50\nactual_minutes: 60"},
		{ID: "2026-11-12", Text: "estimated_minutes: 50\nactual_minutes: 70"},
		{ID: "2026-11-11", Text: "comment: no numbers"},
	}
	if answer := ask(t, NewAgent(w.actions()), "How is the drive?"); !strings.Contains(answer, "Leave 20 minutes earlier") || !strings.Contains(answer, "This includes 15 extra minutes") {
		t.Errorf("feedback: answer = %q, want 15 extra minutes", answer)
	}
}

// TestCheck checks the pickup checks: with no route, the user gets an alert and
// the next answer offers to delegate the pickup. The alert is sent again only
// when the level changes. Without a plan, a check fails.
func TestCheck(t *testing.T) {
	w := newWorld(0)
	w.route = ""
	ag := NewAgent(w.actions())
	if err := ag.Check(t.Context()); err != nil {
		t.Fatal(err)
	}
	if len(w.sent) != 1 || !strings.HasPrefix(w.sent[0], "Alice: Pickup check: Commute agent: The route is not available") || !strings.Contains(w.sent[0], "Send a message") {
		t.Fatalf("sent = %q, want one alert to Alice", w.sent)
	}
	if answer := ask(t, ag, "hi"); !strings.HasSuffix(answer, "because the route is not available?** Reply yes or no.") {
		t.Errorf("answer = %q, want the question", answer)
	}
	ag.Check(t.Context())
	w.route = testRoute(0)
	ag.Check(t.Context())
	if len(w.sent) != 2 || !strings.Contains(w.sent[1], "No delay is expected") {
		t.Errorf("sent = %q, want one more alert for the new level", w.sent)
	}

	w.plan = ""
	if err := ag.Check(t.Context()); err == nil {
		t.Error("check without a plan: no error, want one")
	}
}

// TestWatch checks that the scheduled checks run at their times, and that the
// agent asks for feedback after the estimated arrival, once.
func TestWatch(t *testing.T) {
	w := newWorld(0)
	w.stop = time.Date(2026, 11, 16, 11, 30, 0, 0, time.UTC)
	a := w.actions()
	s := &pickupState{sent: map[string]bool{}, level: map[string]int{}, estimate: map[string]int{"2026-11-16": 50}, feedbackAsked: map[string]bool{}, asked: map[string]string{}}
	var checks []string
	watch(t.Context(), a, s, func(context.Context) error {
		checks = append(checks, w.now.Format("15:04"))
		return errors.New("ignored")
	})
	want := []string{"08:30", "09:30", "09:35", "09:40", "09:45", "09:50", "09:55", "10:00"}
	if !slices.Equal(checks, want) {
		t.Errorf("checks = %v, want %v", checks, want)
	}
	alerts := s.recent(w.now)
	if len(alerts) != 1 || alerts[0].Kind != "feedback" || !strings.Contains(alerts[0].Text, "The estimate was 50 minutes") {
		t.Fatalf("alerts = %v, want one request for feedback", alerts)
	}
	if !alerts[0].Created.Equal(time.Date(2026, 11, 16, 11, 5, 0, 0, time.UTC)) {
		t.Errorf("feedback asked at %v, want 11:05", alerts[0].Created)
	}
	if len(w.sent) != 1 || !strings.HasPrefix(w.sent[0], "Alice: Trip feedback:") {
		t.Errorf("sent = %q, want one request for feedback to Alice", w.sent)
	}
	if len(s.recent(w.now.Add(25*time.Hour))) != 0 {
		t.Error("an alert older than a day is still in the list")
	}
}

// TestFeedback checks the answers to a request for feedback: each value that is
// not valid is refused, and a valid answer saves the trip once.
func TestFeedback(t *testing.T) {
	w := newWorld(0)
	ag := NewAgent(w.actions())
	s := &pickupState{}
	s.add(&pickupAlert{Alert: Alert{Kind: "feedback", Created: w.now}, day: "2026-11-16", estimate: 50})
	a := w.actions()
	for _, tt := range []struct {
		id              string
		actual, rating  int
		comment, errHas string
	}{
		{"1", 0, 5, "", "more than 0"},
		{"1", 60, 6, "", "1 to 5"},
		{"1", 60, 2, " ", "what went wrong"},
		{"9", 60, 5, "", "no feedback alert"},
	} {
		if _, err := s.feedback(t.Context(), a, tt.id, tt.actual, tt.rating, tt.comment); err == nil || !strings.Contains(err.Error(), tt.errHas) {
			t.Errorf("feedback(%v): error = %v, want %q", tt, err, tt.errHas)
		}
	}
	alert, err := s.feedback(t.Context(), a, "1", 65, 2, "Too\nslow")
	if err != nil {
		t.Fatal(err)
	}
	if alert.Status != "answered" || len(w.saved) != 1 || w.saved[0].Text != "date: 2026-11-16\nestimated_minutes: 50\nactual_minutes: 65\nrating: 2\ncomment: Too slow" {
		t.Errorf("alert = %v, saved = %v", alert, w.saved)
	}
	if _, err := s.feedback(t.Context(), a, "1", 65, 4, ""); err == nil {
		t.Error("second answer: no error, want one")
	}
	if _, err := ag.Feedback(t.Context(), "1", 60, 5, ""); err == nil {
		t.Error("unknown alert: no error, want one")
	}
	if len(ag.Alerts(t.Context())) != 0 {
		t.Error("a new agent has alerts")
	}
}

// TestReadOnly checks that the memory cannot change in a conversation that has
// news text from a tool, so that an article cannot change it.
func TestReadOnly(t *testing.T) {
	w := newWorld(0)
	a := w.actions()
	replies := []Message{
		{Role: "assistant", ToolCalls: []ToolCall{{Name: "search_saved_data"}}},
		{Role: "assistant", ToolCalls: []ToolCall{{Name: "save_memory", Arguments: json.RawMessage(`{}`)}}},
		{Role: "assistant", Content: "done"},
	}
	a.Chat = func(context.Context, []Message, []Tool, bool) (Message, error) {
		r := replies[0]
		replies = replies[1:]
		return r, nil
	}
	turn, err := NewAgent(a).Respond(t.Context(), "c", "What did the news say?", func(context.Context, Message) {})
	if err != nil {
		t.Fatal(err)
	}
	i := slices.IndexFunc(turn, func(m Message) bool { return m.ToolName == "save_memory" })
	if i < 0 || !strings.HasPrefix(turn[i].Content, ToolFailed+"the memory cannot change") {
		t.Errorf("save_memory result = %v, want a failure", turn)
	}
}

// TestRun checks that Run refuses thresholds that are not in order, and that it
// gives the agent to the server and stops the pickup checks with the context.
func TestRun(t *testing.T) {
	w := newWorld(0)
	a := w.actions()
	a.MinorDelay, a.MajorDelay = 30, 15
	if err := Run(t.Context(), a); err == nil {
		t.Error("thresholds not in order: no error, want one")
	}

	ctx, cancel := context.WithCancel(t.Context())
	a = w.actions()
	served := false
	a.Serve = func(_ context.Context, ag Agent) error {
		served = ag.Respond != nil && ag.Alerts != nil && ag.Feedback != nil
		cancel()
		return nil
	}
	if err := Run(ctx, a); err != nil || !served {
		t.Errorf("Run: error = %v, served = %v", err, served)
	}
}

// TestHelpers checks the helpers of the coordinator with input that is not
// valid, and the parts of a plan that have defaults.
func TestHelpers(t *testing.T) {
	bad := Message{Role: "tool", ToolName: routeTool, Content: "not JSON"}
	if got := conditionsRequest("hi", bad); got != "Request of the user: hi\n\nRoute: not JSON" {
		t.Errorf("conditionsRequest = %q", got)
	}
	if got, details := splitRoute(bad); !reflect.DeepEqual(got, bad) || details != "" {
		t.Errorf("splitRoute = %v, %q", got, details)
	}
	if _, ok := lastRoute([]Message{bad}); ok {
		t.Error("lastRoute found a route that is not JSON")
	}
	if got := mentions([]Message{bad, {Role: "tool", Content: `{"mention":"M"}`}}); got != "\n\nM" {
		t.Errorf("mentions = %q", got)
	}
	if got := otherRoutes(map[string]any{}); got != "a different route" {
		t.Errorf("otherRoutes = %q", got)
	}
	if got := (pickupPlan{}).user(); got != "the user" {
		t.Errorf("user = %q", got)
	}
	if _, ok := leaveTime(pickupPlan{leaveBy: "late"}, time.Now()); ok {
		t.Error("leaveTime accepted late")
	}
	if got := startResults(t.Context(), Actions{}, new(int), func(context.Context, Message) {}); got != nil {
		t.Errorf("startResults without the start tools = %v", got)
	}
	s := &pickupState{asked: map[string]string{}}
	if got := s.answer(t.Context(), Actions{}, "1", true); got != `I did not send a pickup request, because no delegate alert has the ID "1".` {
		t.Errorf("answer = %q", got)
	}
}

// TestFailures checks that an empty answer gets a readiness check, and that a
// memory or a trip store that fails gives no plan and no extra minutes.
func TestFailures(t *testing.T) {
	w := newWorld(0)
	a := w.actions()
	replies := []string{"", "hello"}
	a.Chat = func(_ context.Context, messages []Message, _ []Tool, _ bool) (Message, error) {
		r := replies[0]
		replies = replies[1:]
		return Message{Role: "assistant", Content: r}, nil
	}
	turn, err := NewAgent(a).Respond(t.Context(), "c", "hi", func(context.Context, Message) {})
	if err != nil || turn[len(turn)-1].Content != "hello" || !slices.ContainsFunc(turn, func(m Message) bool { return strings.HasPrefix(m.Content, NotReady+"the answer is empty") }) {
		t.Errorf("turn = %v, error = %v, want a readiness check and then hello", turn, err)
	}

	for _, out := range []string{"", "not JSON"} {
		w.plan = ""
		a = w.actions()
		a.Planner[0].Run = func(context.Context, json.RawMessage) (string, error) {
			if out == "" {
				return "", errors.New("broken")
			}
			return out, nil
		}
		if _, ok := readPlan(t.Context(), a); ok {
			t.Errorf("readPlan with %q found a plan", out)
		}
	}
	if _, ok := readPlan(t.Context(), Actions{}); ok {
		t.Error("readPlan without a memory tool found a plan")
	}

	a.Trips = func(context.Context, string, string, string, int) ([]Document, error) {
		return nil, errors.New("broken")
	}
	if got := refinement(t.Context(), a); got != 0 {
		t.Errorf("refinement with a broken store = %d, want 0", got)
	}
}
