package providers

import (
	"context"
	"io"
	"log/slog"

	"github.com/flying-robot/commute-agent/internal"
)

// NewSlogProvider returns an internal.LogFunc that writes each event as one
// JSON line to the provided writer. This ensures that output from the agent
// is structured and readily-parsed by other tools. The handler uses its default
// minimum level, so Debug events are not written.
func NewSlogProvider(w io.Writer) internal.LogFunc {
	log := slog.New(slog.NewJSONHandler(w, nil))
	return func(ctx context.Context, level internal.Level, event string, attrs ...any) {
		log.Log(ctx, toSlogLevel(level), event, attrs...)
	}
}

// toSlogLevel converts an internal.Level to the matching slog.Level. An unknown
// level becomes slog.LevelInfo, so that the event is not lost.
func toSlogLevel(level internal.Level) slog.Level {
	switch level {
	case internal.LevelDebug:
		return slog.LevelDebug
	case internal.LevelWarn:
		return slog.LevelWarn
	case internal.LevelError:
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
