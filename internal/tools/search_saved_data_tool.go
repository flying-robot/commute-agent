package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// savedResults is the number of documents that one search gives.
const savedResults = 5

// savedCollections are the collections of the vector store that the model can
// search, with the tool that gets new data for each one.
var savedCollections = map[string]string{
	newsCollection:    "get_current_news",
	weatherCollection: "get_route_weather",
}

// SearchSavedData returns a tool that searches the news and the weather that
// the other tools saved in the vector store, by meaning. The data expires after
// a short time, so the tool finds only data that is still new enough to use. The
// model can then answer without a new request to WTOP or the National Weather
// Service.
func SearchSavedData(search internal.SearchFunc) internal.Tool {
	names := slices.Sorted(maps.Keys(savedCollections))

	return internal.Tool{
		Name:        "search_saved_data",
		Description: "Search the news and the weather that were checked in the last few minutes, by meaning. Each result says how many minutes ago it was checked. Old data is deleted, so if the search finds nothing for the places that you need, get new data with the get_current_news or get_route_weather tool.",
		Parameters: []internal.Parameter{{
			Name:        "collection",
			Type:        "string",
			Description: "The kind of data to search.",
			Enum:        names,
			Required:    true,
		}, {
			Name:        "query",
			Type:        "string",
			Description: "What to find, for example: weather at Bethesda, MD, or: news about I-495 and Arlington, VA.",
			Required:    true,
		}},
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments, and make sure that the collection exists.
			in, err := decode(args)
			if err != nil {
				return "", err
			}
			collection, query := get[string](in, "collection"), get[string](in, "query")
			tool, ok := savedCollections[collection]
			if !ok {
				return "", fmt.Errorf("collection %q is not one of %v", collection, names)
			}
			if query == "" {
				return "", fmt.Errorf("query is required")
			}

			// Next, find the nearest saved documents, with the age of each one in
			// minutes, so that the model can tell the user how old the data is.
			docs, err := search(ctx, collection, "", query, savedResults)
			if err != nil {
				return "", fmt.Errorf("failed to search the saved data: %w", err)
			}
			results := []map[string]any{}
			for _, d := range docs {
				results = append(results, map[string]any{
					"text":                d.Text,
					"checked_minutes_ago": int(time.Since(d.SavedAt).Minutes()),
				})
			}

			// Finally, return the results as JSON. If there are none, say which tool
			// gets new data, so that the model does not stop.
			out := map[string]any{"collection": collection, "results": results}
			if len(results) == 0 {
				out["note"] = fmt.Sprintf("Nothing new enough is saved. Call the %s tool to get new data.", tool)
			}
			return toJSON(out)
		},
	}
}
