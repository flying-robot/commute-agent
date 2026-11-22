package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/flying-robot/commute-agent/internal"
)

// ForgetMemory returns a tool that deletes a fact that the user asked the agent
// to remember, by its name. It is a separate tool from save_memory, so that the
// model cannot delete a fact by mistake while it saves one.
func ForgetMemory(del internal.DeleteFunc) internal.Tool {
	return internal.Tool{
		Name:        "forget_memory",
		Description: "Forget a fact that the user asked you to remember, by its name. Use it only when the user asks you to forget something.",
		Parameters: []internal.Parameter{{
			Name:        "name",
			Type:        "string",
			Description: "The name of the fact to forget, for example trip to work. Use the search_memory tool to find the saved names.",
			Required:    true,
		}},
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments. A name is compared without case, in the
			// same way as in save_memory.
			in, err := decode(args)
			if err != nil {
				return "", err
			}
			name := strings.ToLower(strings.TrimSpace(get[string](in, "name")))
			if name == "" {
				return "", errors.New("a name is required to forget a fact")
			}
			if name == internal.PickupFact {
				return "", errors.New("the pickup plan can change only in the memory file. Tell the user to edit it there")
			}

			// Finally, delete the fact. If there is no fact with that name, say how
			// to find the saved names, so that the model can try again.
			ok, err := del(ctx, memoryCollection, name)
			if err != nil {
				return "", fmt.Errorf("failed to forget the fact: %w", err)
			}
			if !ok {
				return "", fmt.Errorf("nothing is saved as %q, use the search_memory tool to find the saved names", name)
			}
			return toJSON(map[string]string{"forgot": name})
		},
	}
}
