package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/flying-robot/commute-agent/internal"
)

// memoryCollection is the collection of the store that has the facts that the
// user asks the agent to remember.
const memoryCollection = "memory"

// SaveMemory returns a tool that saves a fact that the user asks the agent to
// remember, such as a named trip like "trip to work". The name is the ID of the
// fact, so that a fact with the same name is replaced. The saved text starts
// with the name, so that a search for the name finds the fact.
func SaveMemory(save internal.SaveFunc) internal.Tool {
	return internal.Tool{
		Name:        "save_memory",
		Description: "Save a fact that the user asks you to remember, such as a named trip (for example \"trip to work\") with its start, destination, and usual departure time. A fact with the same name is replaced.",
		Parameters: []internal.Parameter{{
			Name:        "name",
			Type:        "string",
			Description: "The name of the fact, for example trip to work.",
			Required:    true,
		}, {
			Name:        "details",
			Type:        "string",
			Description: "The fact to save, in full, for example: start 11725 Berry Rd, Waldorf, MD; destination Union Station, Washington, DC; leaves Monday to Friday at 8:00.",
			Required:    true,
		}},
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments. A name is compared without case, so that
			// "Trip to Work" and "trip to work" are the same fact.
			in, err := decode(args)
			if err != nil {
				return "", err
			}
			name := strings.ToLower(strings.TrimSpace(get[string](in, "name")))
			details := strings.TrimSpace(get[string](in, "details"))
			if name == "" || details == "" {
				return "", errors.New("a name and details are required to save a fact")
			}
			if name == internal.PickupFact {
				return "", errors.New("the pickup plan can change only in the memory file. Tell the user to edit it there")
			}

			// Finally, save the fact, and say what was saved. A fact never expires.
			doc := internal.Document{ID: name, Text: name + ": " + details}
			if err := save(ctx, memoryCollection, []internal.Document{doc}, 0); err != nil {
				return "", fmt.Errorf("failed to save the fact: %w", err)
			}
			return toJSON(map[string]string{"saved": name})
		},
	}
}
