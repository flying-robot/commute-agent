package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/flying-robot/commute-agent/internal"
	"github.com/ollama/ollama/api"
)

// keepAlive is how long Ollama keeps a model in memory after its last request.
// The default is 5 minutes, and a commute agent is often used once and then
// not again for some time, so each first question would load the model again.
var keepAlive = &api.Duration{Duration: time.Hour}

// NewOllamaProvider returns a ChatFunc that can interact with the specified
// Ollama model located at OLLAMA_HOST. It acts as an adapter that translates
// between our internal format and the format that the Ollama package uses.
func NewOllamaProvider(model string) (internal.ChatFunc, error) {
	// This client will be captured by the returned ChatFunc.
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}
	if model == "" {
		return nil, fmt.Errorf("model must be specified")
	}

	return func(ctx context.Context, messages []internal.Message, tools []internal.Tool, think bool) (internal.Message, error) {
		// First, convert the internal messages to the Ollama format.
		msgs, err := toOllamaMessages(messages)
		if err != nil {
			return internal.Message{}, fmt.Errorf("failed to convert messages to Ollama format: %w", err)
		}

		// Next, create a response structure from the messages. The request won't use
		// streaming, but note that the field requires a pointer to a boolean. Thinking
		// is set explicitly, so that it does not depend on the default of the model
		// or Ollama.
		req := api.ChatRequest{
			Model:     model,
			Messages:  msgs,
			Tools:     toOllamaTools(tools),
			Stream:    new(false),
			Think:     &api.ThinkValue{Value: think},
			KeepAlive: keepAlive,
		}

		// Next, capture the reply via a callback from the Ollama package.
		var reply api.Message
		err = client.Chat(ctx, &req, func(resp api.ChatResponse) error {
			reply = resp.Message
			return nil
		})
		if err != nil {
			return internal.Message{}, fmt.Errorf("failed to get chat response from Ollama: %w", err)
		}

		// Finally, convert the Ollama message back into the internal format.
		return fromOllamaMessage(reply)
	}, nil
}

// NewOllamaEmbedProvider returns an internal.EmbedFunc that embeds texts with the
// specified Ollama embedding model located at OLLAMA_HOST.
//
// The nomic-embed-text models find related texts well only when each text
// starts with its task, "search_query: " or "search_document: ", so the
// provider adds the task for those models.
func NewOllamaEmbedProvider(model string) (internal.EmbedFunc, error) {
	// This client will be captured by the returned EmbedFunc.
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}
	if model == "" {
		return nil, fmt.Errorf("embedding model must be specified")
	}

	return func(ctx context.Context, texts []string, query bool) ([][]float32, error) {
		if strings.HasPrefix(model, "nomic-embed-text") {
			task := "search_document: "
			if query {
				task = "search_query: "
			}
			prefixed := make([]string, len(texts))
			for i, t := range texts {
				prefixed[i] = task + t
			}
			texts = prefixed
		}
		resp, err := client.Embed(ctx, &api.EmbedRequest{Model: model, Input: texts, KeepAlive: keepAlive})
		if err != nil {
			return nil, fmt.Errorf("failed to get embeddings from Ollama: %w", err)
		}
		return resp.Embeddings, nil
	}, nil
}

// toOllamaMessages converts a slice of internal.Message to a slice of api.Message
// that can be given to the Ollama API. Any tool calls within the message will be
// unmarshaled into the appropriate format as well.
func toOllamaMessages(messages []internal.Message) ([]api.Message, error) {
	out := make([]api.Message, len(messages))

	for i, m := range messages {
		msg := api.Message{
			Role:     m.Role,
			Content:  m.Content,
			Thinking: m.Thinking,
			ToolName: m.ToolName,
		}

		for _, c := range m.ToolCalls {
			args := api.NewToolCallFunctionArguments()
			if err := json.Unmarshal(c.Arguments, &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool call arguments: %w", err)
			}
			msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      c.Name,
					Arguments: args,
				},
			})
		}
		out[i] = msg
	}

	return out, nil
}

// fromOllamaMessage converts an api.Message from the Ollama API into an internal.Message
// that the application can use. Any tool calls within the message will be marshaled into
// the appropriate format as well.
//
// The model sometimes puts the end of its thinking in the content, followed by
// a "</think>" tag. The text before the tag is moved to the thinking, so that
// the user sees only the answer and the log still has all of the reasoning.
func fromOllamaMessage(m api.Message) (internal.Message, error) {
	msg := internal.Message{
		Role:     m.Role,
		Content:  m.Content,
		Thinking: m.Thinking,
		ToolName: m.ToolName,
	}
	if i := strings.LastIndex(m.Content, "</think>"); i >= 0 {
		msg.Thinking += m.Content[:i]
		msg.Content = strings.TrimSpace(m.Content[i+len("</think>"):])
	}

	for _, c := range m.ToolCalls {
		args, err := json.Marshal(c.Function.Arguments)
		if err != nil {
			return internal.Message{}, fmt.Errorf("failed to marshal tool call arguments: %w", err)
		}
		msg.ToolCalls = append(msg.ToolCalls, internal.ToolCall{
			Name:      c.Function.Name,
			Arguments: args,
		})
	}
	return msg, nil
}

// toOllamaTools converts a slice of internal.Tool to a slice of api.Tool that can be
// given to the Ollama API. The Ollama package uses ordered maps and a variety of specific
// functions that need to be invoked to correctly assemble a tool definition.
func toOllamaTools(tools []internal.Tool) []api.Tool {
	out := make([]api.Tool, len(tools))

	for i, t := range tools {
		props := api.NewToolPropertiesMap()
		required := make([]string, 0)

		for _, p := range t.Parameters {
			prop := api.ToolProperty{
				Type:        api.PropertyType{p.Type},
				Description: p.Description,
			}
			for _, e := range p.Enum {
				prop.Enum = append(prop.Enum, e)
			}
			props.Set(p.Name, prop)
			if p.Required {
				required = append(required, p.Name)
			}
		}

		out[i] = api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters: api.ToolFunctionParameters{
					Type:       "object",
					Properties: props,
					Required:   required,
				},
			},
		}
	}

	return out
}
