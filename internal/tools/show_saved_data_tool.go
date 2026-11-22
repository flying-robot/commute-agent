package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// shownLength is the largest number of characters of a document that the list
// shows, so that a long article does not fill the chat.
const shownLength = 140

// ShowSavedData returns a tool that shows the contents of the vector store as a
// Markdown list, so that the user can see which news and weather the agent saved
// and how old each one is. The tool writes the list itself, so that the model
// only has to show it.
func ShowSavedData(search internal.SearchFunc) internal.Tool {
	names := slices.Sorted(maps.Keys(savedCollections))

	return internal.Tool{
		Name:        "show_saved_data",
		Description: "Show the user the news and the weather that are saved now, with the age of each one. Use it when the user asks what is saved or stored. Show the markdown of the result to the user exactly as it is.",
		Parameters: []internal.Parameter{{
			Name:        "collection",
			Type:        "string",
			Description: "The kind of data to show. Leave it out to show all of it.",
			Enum:        names,
		}},
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments, and choose the collections to show.
			in, err := decode(args)
			if err != nil {
				return "", err
			}
			collections := names
			if c := get[string](in, "collection"); c != "" {
				if _, ok := savedCollections[c]; !ok {
					return "", fmt.Errorf("collection %q is not one of %v", c, names)
				}
				collections = []string{c}
			}

			// Next, list each collection from the newest document, with one short
			// line for each document. An empty query lists all the documents.
			var md strings.Builder
			for _, c := range collections {
				docs, err := search(ctx, c, "", "", math.MaxInt)
				if err != nil {
					return "", fmt.Errorf("failed to list the saved %s: %w", c, err)
				}
				fmt.Fprintf(&md, "**Saved %s** (%d)\n\n", c, len(docs))
				if len(docs) == 0 {
					md.WriteString("- Nothing is saved.\n")
				}
				for _, d := range docs {
					fmt.Fprintf(&md, "- %s: %s\n", ago(d.SavedAt), summary(d.Text))
				}
				md.WriteString("\n")
			}

			// Finally, return the list as JSON.
			return toJSON(map[string]any{"markdown": strings.TrimSpace(md.String())})
		},
	}
}

// ago returns how long ago a time was, in whole minutes.
func ago(t time.Time) string {
	switch m := int(time.Since(t).Minutes()); m {
	case 0:
		return "less than 1 minute ago"
	case 1:
		return "1 minute ago"
	default:
		return fmt.Sprintf("%d minutes ago", m)
	}
}

// summary returns one short line of a document. The weather is saved as JSON,
// so its line is the place, the time, and the forecast. Other text gives its
// first line, without the "Title: " label of an article.
func summary(text string) string {
	var w map[string]any
	if json.Unmarshal([]byte(text), &w) == nil && get[string](w, "place") != "" {
		forecast := "no forecast"
		if f := get[string](w, "weather", "forecast"); f != "" {
			forecast = fmt.Sprintf("%s, %.0f°F, %d alerts", f, get[float64](w, "weather", "temperature_f"), len(get[[]any](w, "weather", "alerts")))
		}
		text = fmt.Sprintf("%s at %s: %s", get[string](w, "place"), get[string](w, "time"), forecast)
	}
	line, _, _ := strings.Cut(text, "\n")
	line = strings.TrimPrefix(line, "Title: ")
	if r := []rune(line); len(r) > shownLength {
		line = string(r[:shownLength]) + "..."
	}
	return line
}
