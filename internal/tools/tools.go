package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// userAgent names the agent in each request to OpenStreetMap, the National
// Weather Service, and WTOP, because they ask each app to name itself.
const userAgent = "commute-agent (personal use)"

// client sends the requests of the tools. Its timeout stops a service that does
// not answer from holding a turn. It is long, because a WTOP search that WTOP
// has not done before can take more than 20 seconds. retryDelay is the wait
// before the one retry of a request that failed for a reason that can pass.
var client = &http.Client{Timeout: 60 * time.Second}

const retryDelay = time.Second

// All returns the tools of the trip planner and of the conditions agent. The
// route and weather tools use the TomTom API key, the news and weather tools
// save their data in the vector store for ttl, the news tool gives up to
// newsResults articles, and the memory tools use the store of the facts.
func All(apiKey string, save internal.SaveFunc, search internal.SearchFunc, ttl time.Duration, newsResults int, remember internal.SaveFunc, recall internal.SearchFunc, forget internal.DeleteFunc) (planner, conditions []internal.Tool) {
	planner = []internal.Tool{
		GetCurrentTime(),
		GetRoute(apiKey),
		SearchSavedData(search),
		ShowSavedData(search),
		SaveMemory(remember),
		SearchMemory(recall),
		ForgetMemory(forget),
	}
	conditions = []internal.Tool{
		GetRouteWeather(apiKey, save, search, ttl),
		GetCurrentNews(save, search, ttl, newsResults),
	}
	return planner, conditions
}

// decode returns the JSON arguments of a tool call as a map. No arguments give an
// empty map, because the model can send none when all the arguments are optional.
func decode(args json.RawMessage) (map[string]any, error) {
	in := map[string]any{}
	if len(args) > 0 {
		if err := json.Unmarshal(args, &in); err != nil {
			return nil, fmt.Errorf("failed to unmarshal arguments: %w", err)
		}
	}
	return in, nil
}

// toJSON returns v as the JSON text of a tool result.
func toJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	return string(b), err
}

// fetch sends a GET request with the user agent, and with the Accept header if it
// is not empty, and returns the body. A status other than 200 is an error. The
// context cancels the request if the caller of the agent disconnects.
func fetch(ctx context.Context, url, accept string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	resp, err := do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// do sends a GET request, and sends it one more time after retryDelay if it
// failed for a reason that can pass: a network error, too many requests, or a
// server error. A timeout is not sent again, because a second request to a slow
// service is also slow. A GET request has no body, so it can be sent again as
// it is.
func do(req *http.Request) (*http.Response, error) {
	resp, err := client.Do(req)
	if err == nil && resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode < 500 {
		return resp, nil
	}
	if ne, ok := errors.AsType[net.Error](err); ok && ne.Timeout() {
		return nil, err
	}
	if err == nil {
		resp.Body.Close()
	}
	select {
	case <-req.Context().Done():
		return nil, req.Context().Err()
	case <-time.After(retryDelay):
	}
	return client.Do(req)
}
