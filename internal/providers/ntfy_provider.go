package providers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// ntfyURL is the base URL of the public ntfy.sh server, and ntfyTimeout is how
// long a publish can take.
const (
	ntfyURL     = "https://ntfy.sh/"
	ntfyTimeout = 10 * time.Second
)

// NewNtfyProvider returns an internal.NotifyFunc that publishes each
// notification to a topic on ntfy.sh, so that each phone that subscribes to the
// topic in the ntfy app gets it. ntfy.sh is free and needs no account, but
// anyone who knows the topic can read it, so the topic must be hard to guess.
// All the people share the topic, so the title says whom a notification is for.
//
// fail injects a failure into each send instead of a publish, to test the
// agent:
//   - "error" fails as a server error does.
//   - "timeout" waits ntfyTimeout, or until the context is canceled, and then
//     fails with internal.ErrNoResponse.
//   - "rate_limit" fails as ntfy.sh does when there are too many requests.
func NewNtfyProvider(topic, fail string) (internal.NotifyFunc, error) {
	if topic == "" {
		return nil, errors.New("the ntfy.sh topic is not set")
	}
	switch fail {
	case "", "error", "timeout", "rate_limit":
	default:
		return nil, fmt.Errorf("unknown failure %q, want error, timeout, or rate_limit", fail)
	}
	client := &http.Client{Timeout: ntfyTimeout}

	return func(ctx context.Context, to, title, body string) (string, error) {
		// First, inject the failure, if there is one.
		switch fail {
		case "error":
			return "", errors.New("ntfy.sh: status 500: internal server error")
		case "timeout":
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(ntfyTimeout):
				return "", fmt.Errorf("ntfy.sh: %w after %s", internal.ErrNoResponse, ntfyTimeout)
			}
		case "rate_limit":
			return "", errors.New("ntfy.sh: status 429: too many requests")
		}

		// Next, publish the body to the topic, with the title in its header. A
		// timeout gets ErrNoResponse, because the notification can still arrive.
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, ntfyURL+url.PathEscape(topic), strings.NewReader(body))
		if err != nil {
			return "", err
		}
		req.Header.Set("Title", "For "+to+": "+title)
		resp, err := client.Do(req)
		if err != nil {
			if ne, ok := errors.AsType[net.Error](err); ok && ne.Timeout() {
				return "", fmt.Errorf("ntfy.sh: %w: %v", internal.ErrNoResponse, err)
			}
			return "", fmt.Errorf("ntfy.sh: %w", err)
		}
		defer resp.Body.Close()

		// Finally, return the ID of the notification. A status other than 200 is
		// an error, with the start of the body, which says why.
		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
			return "", fmt.Errorf("ntfy.sh: status %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
		}
		var out struct {
			ID string `json:"id"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			return "", fmt.Errorf("ntfy.sh: failed to read the response: %w", err)
		}
		return out.ID, nil
	}, nil
}
