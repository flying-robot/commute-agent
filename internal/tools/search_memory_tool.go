package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	"github.com/flying-robot/commute-agent/internal"
)

// memoryResults is the number of facts that a search with a query gives.
const memoryResults = 5

// SearchMemory returns a tool that finds the facts that the user asked the agent
// to remember. Without a query it gives all the facts, because there are few of
// them, and the model can then choose the one that the user means. A search by
// meaning can miss a fact, so all the facts are the safer choice.
func SearchMemory(search internal.SearchFunc) internal.Tool {
	return internal.Tool{
		Name:        "search_memory",
		Description: "Find the facts that the user asked you to remember. Use it when a question names a saved trip or says \"my commute\", \"work\", or \"home\", and use the saved places and times. Leave the query out to get all the facts, which is best when you are not sure which fact the user means.",
		Parameters: []internal.Parameter{{
			Name:        "query",
			Type:        "string",
			Description: "What to find, for example trip to work. Leave it out to get all the facts.",
		}},
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments. The model can send no arguments at all,
			// because the query is optional.
			in, err := decode(args)
			if err != nil {
				return "", err
			}
			query := get[string](in, "query")
			n := math.MaxInt
			if query != "" {
				n = memoryResults
			}

			// Finally, find the facts, and return each one with its name.
			docs, err := search(ctx, memoryCollection, "", query, n)
			if err != nil {
				return "", fmt.Errorf("failed to search the facts: %w", err)
			}
			memories := []map[string]string{}
			for _, d := range docs {
				memories = append(memories, map[string]string{"name": d.ID, "fact": d.Text})
			}
			return toJSON(map[string]any{"memories": memories})
		},
	}
}
