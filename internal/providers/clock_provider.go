package providers

import (
	"context"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// NewClockProvider returns an internal.NowFunc that reads the system clock. The
// agent uses it to measure how long each model call and tool call takes, and to
// find when the next pickup check is.
func NewClockProvider() internal.NowFunc {
	return func(context.Context) time.Time {
		return time.Now()
	}
}

// NewSleepProvider returns an internal.SleepFunc that waits on the system clock.
func NewSleepProvider() internal.SleepFunc {
	return func(ctx context.Context, d time.Duration) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(d):
			return nil
		}
	}
}
