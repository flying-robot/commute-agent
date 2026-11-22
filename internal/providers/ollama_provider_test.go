package providers

import (
	"encoding/json"
	"testing"

	"github.com/flying-robot/commute-agent/internal"
)

// TestNewOllamaProviders checks that both providers need a model.
func TestNewOllamaProviders(t *testing.T) {
	if _, err := NewOllamaProvider(""); err == nil {
		t.Error("chat with no model: no error, want one")
	}
	if _, err := NewOllamaEmbedProvider(""); err == nil {
		t.Error("embedding with no model: no error, want one")
	}
	if _, err := NewOllamaProvider("model"); err != nil {
		t.Errorf("chat with a model: %v, want no error", err)
	}
	if _, err := NewOllamaEmbedProvider("model"); err != nil {
		t.Errorf("embedding with a model: %v, want no error", err)
	}
}

// TestOllamaMessages checks that a message with tool calls goes to the Ollama
// format and back without a change, and that arguments that are not JSON are an
// error.
func TestOllamaMessages(t *testing.T) {
	in := internal.Message{Role: "assistant", Content: "c", Thinking: "t", ToolName: "n", ToolCalls: []internal.ToolCall{
		{Name: "get_route", Arguments: json.RawMessage(`{"start":"A"}`)},
	}}
	out, err := toOllamaMessages([]internal.Message{in})
	if err != nil {
		t.Fatal(err)
	}
	back, err := fromOllamaMessage(out[0])
	if err != nil {
		t.Fatal(err)
	}
	if back.Role != in.Role || back.Content != in.Content || back.Thinking != in.Thinking || back.ToolName != in.ToolName ||
		len(back.ToolCalls) != 1 || back.ToolCalls[0].Name != "get_route" || string(back.ToolCalls[0].Arguments) != `{"start":"A"}` {
		t.Errorf("message after the round trip = %+v", back)
	}

	bad := internal.Message{ToolCalls: []internal.ToolCall{{Name: "x", Arguments: json.RawMessage(`not json`)}}}
	if _, err := toOllamaMessages([]internal.Message{bad}); err == nil {
		t.Error("arguments that are not JSON: no error, want one")
	}
}

// TestFromOllamaMessageThink checks that the text before a "</think>" tag in the
// content moves to the thinking.
func TestFromOllamaMessageThink(t *testing.T) {
	out, _ := toOllamaMessages([]internal.Message{{Role: "assistant", Content: "plan</think>  answer", Thinking: "first "}})
	got, err := fromOllamaMessage(out[0])
	if err != nil || got.Content != "answer" || got.Thinking != "first plan" {
		t.Errorf("message = %+v, %v, want the answer and all the thinking", got, err)
	}
}

// TestToOllamaTools checks the type, the description, the enum, and the
// required parameters of a tool.
func TestToOllamaTools(t *testing.T) {
	tools := toOllamaTools([]internal.Tool{{Name: "get_route", Description: "d", Parameters: []internal.Parameter{
		{Name: "start", Type: "string", Required: true},
		{Name: "avoid", Type: "string", Enum: []string{"tollRoads"}},
	}}})
	f := tools[0].Function
	start, _ := f.Parameters.Properties.Get("start")
	avoid, _ := f.Parameters.Properties.Get("avoid")
	if tools[0].Type != "function" || f.Name != "get_route" || f.Description != "d" ||
		len(f.Parameters.Required) != 1 || f.Parameters.Required[0] != "start" ||
		start.Type[0] != "string" || len(avoid.Enum) != 1 || avoid.Enum[0] != "tollRoads" {
		t.Errorf("tool = %+v", tools[0])
	}
}
