package internal

import (
	"context"
	"encoding/json"
	"errors"
	"time"
)

// The levels of a logged event, from the least to the most severe.
const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

// A Message contains a single exchange within a larger conversation.
type Message struct {
	Role      string        `json:"role"`
	Content   string        `json:"content"`
	Thinking  string        `json:"thinking"`
	ToolCalls []ToolCall    `json:"tool_calls,omitempty"`
	ToolName  string        `json:"tool_name,omitempty"`
	Duration  time.Duration `json:"duration,omitempty"`
}

// A ToolCall describes a single invocation of a tool within a message.
type ToolCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// A Tool is a utility or function that can be invoked by the agent.
type Tool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  []Parameter `json:"parameters"`
	Run         ToolFunc    `json:"run"`
}

// A Parameter describes an input that a tool can accept.
type Parameter struct {
	Name        string   `json:"name"`
	Type        string   `json:"type"`
	Description string   `json:"description"`
	Enum        []string `json:"enum,omitempty"`
	Required    bool     `json:"required"`
}

// A ChatFunc sends the conversation and tools to the model and returns its reply.
// If think is false, the model answers without reasoning first, which is much
// faster for a task that only writes text from given data.
type ChatFunc func(ctx context.Context, messages []Message, tools []Tool, think bool) (Message, error)

// A NowFunc returns the current time.
type NowFunc func(context.Context) time.Time

// A ToolFunc invokes a tool with JSON arguments and returns a text observation.
type ToolFunc func(context.Context, json.RawMessage) (string, error)

// A LogFunc records events at a severity level using key-value pairs.
type LogFunc func(context.Context, Level, string, ...any)

// A MessageFunc gets each message of a turn as soon as the agent has it, so that
// a caller can show the progress of the agent while it works.
type MessageFunc func(context.Context, Message)

// A AgenticLoopFunc answers one message from a caller in a conversation. Each
// conversation ID has its own history, and a new ID starts a new conversation.
// It gives each new reply of the model and each tool result to the MessageFunc,
// and returns the messages of the turn, from the message of the caller to the
// answer, so that the caller can show how the agent got its answer. The last
// message is the answer. If the turn fails, it returns the messages up to the
// failure with the error.
type AgenticLoopFunc func(ctx context.Context, conversation, content string, onMessage MessageFunc) ([]Message, error)

// A ServeFunc serves the agent to the user until the context is canceled.
type ServeFunc func(context.Context, Agent) error

// An Agent is what the server gives to the user. Respond answers a message in a
// conversation, and asks and answers the question whether to delegate the
// pickup. Alerts gives the requests for feedback after a trip, and Feedback
// answers one. Check runs one pickup check at once, and Watch runs the
// scheduled pickup checks until the context is canceled.
type Agent struct {
	Respond  AgenticLoopFunc
	Alerts   AlertsFunc
	Feedback FeedbackFunc
	Check    func(context.Context) error
	Watch    func(context.Context)
}

// An Alert is a decision that waits for the user, or the result of one. Kind is
// "delegate" for an offer to delegate the pickup, which the chat asks about, or
// "feedback" for a request for feedback after a trip. Status is "pending" until
// the user answers.
type Alert struct {
	ID      string    `json:"id"`
	Kind    string    `json:"kind"`
	Text    string    `json:"text"`
	Status  string    `json:"status"`
	Created time.Time `json:"created"`
}

// An AlertsFunc returns the alerts of the last day, from the newest.
type AlertsFunc func(context.Context) []Alert

// A FeedbackFunc records the actual travel time in minutes, a rating from 1 to
// 5, and a comment for a request for feedback, and returns the alert with its
// new status.
type FeedbackFunc func(ctx context.Context, id string, actualMinutes, rating int, comment string) (Alert, error)

// A Document is one text in the vector store. A document replaces the document
// with the same ID in its collection, so that a new check of a feed does not add
// copies. Source is where the text came from, such as a feed, so that a search
// can find only the documents of one source. The store sets SavedAt, so that a
// reader can tell how old the text is. A search sets Score, the similarity of
// the document to the query, so that a reader can drop unrelated documents.
type Document struct {
	ID      string
	Source  string
	Text    string
	SavedAt time.Time
	Score   float32
}

// An EmbedFunc returns one embedding vector for each text. Texts with a similar
// meaning have vectors that are near each other. query is true for the text of
// a search, and false for documents, because some models embed them in
// different ways.
type EmbedFunc func(ctx context.Context, texts []string, query bool) ([][]float32, error)

// A SaveFunc saves documents in a collection of the vector store. If ttl is more
// than zero, the documents are deleted after that time, so that old data must be
// checked again. If ttl is zero, the documents stay.
type SaveFunc func(ctx context.Context, collection string, docs []Document, ttl time.Duration) error

// A SearchFunc returns up to n documents of a collection, from the document whose
// meaning is nearest to the query. If source is not empty, it returns only the
// documents from that source. If query is empty, it returns the documents from
// the newest, with no scores, so that a caller can list the store.
type SearchFunc func(ctx context.Context, collection, source, query string, n int) ([]Document, error)

// A DeleteFunc deletes the document with an ID from a collection of a store. It
// returns false if the collection has no document with that ID.
type DeleteFunc func(ctx context.Context, collection, id string) (bool, error)

// A NotifyFunc sends a notification with a title and a body to a person, by
// name, and returns the ID of the notification.
type NotifyFunc func(ctx context.Context, to, title, body string) (string, error)

// ErrNoResponse is in the error of a send that got no response. The message can
// still arrive, so the agent does not send it again.
var ErrNoResponse = errors.New("no response")

// A SleepFunc waits for a duration, and returns the error of the context if it
// is canceled first.
type SleepFunc func(ctx context.Context, d time.Duration) error

// A Level is the severity of a logged event.
type Level int

// An Actions structure groups impure operations for injection into the pure core.
// Chat is the model of the trip planner, and SmallChat is the smaller and faster
// model of the conditions agent and the reporter, whose tasks are simpler.
// Planner has the tools of the trip planner, and Conditions has the tools of the
// conditions agent. Notify sends the pickup requests and the alerts. SaveTrip and
// Trips keep the feedback after each trip, which refines later estimates. Serve
// gives the agent to the user. MinorDelay and MajorDelay are the thresholds of the
// decider in minutes: above MinorDelay, it offers to delegate the pickup, and above
// MajorDelay, it recommends to delegate it. If ForceDelegation is true, it recommends
// to delegate the pickup for each route, to test the delegation.
type Actions struct {
	Chat            ChatFunc
	SmallChat       ChatFunc
	Log             LogFunc
	Now             NowFunc
	Sleep           SleepFunc
	Planner         []Tool
	Conditions      []Tool
	Notify          NotifyFunc
	SaveTrip        SaveFunc
	Trips           SearchFunc
	Serve           ServeFunc
	MinorDelay      int
	MajorDelay      int
	ForceDelegation bool
}

// PickupFact is the name of the saved fact that has the pickup plan: the trip,
// the times, and the contacts. Only the user can change it, in the memory file,
// so that a model or an article cannot change who gets a pickup request.
const PickupFact = "pickup"
