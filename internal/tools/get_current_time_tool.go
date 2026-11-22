package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// GetCurrentTime returns a tool that gives the current date and time, in local time
// or in a time zone that the model asks for. The tool reads the system clock.
func GetCurrentTime() internal.Tool {
	return internal.Tool{
		Name:        "get_current_time",
		Description: "Get the current date and time.",
		Parameters: []internal.Parameter{{
			Name:        "timezone",
			Type:        "string",
			Description: "An IANA time zone name, for example America/New_York. The default is local time.",
		}},
		Run: func(_ context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments. The model can send no arguments at all,
			// because the time zone is optional.
			in, err := decode(args)
			if err != nil {
				return "", err
			}

			// Next, read the clock, and convert the time to the time zone if the
			// model gave one.
			t := time.Now()
			if tz := get[string](in, "timezone"); tz != "" {
				loc, err := time.LoadLocation(tz)
				if err != nil {
					return "", fmt.Errorf("failed to load time zone: %w", err)
				}
				t = t.In(loc)
			}

			// Next, list the dates of the next seven days with their weekdays and
			// their UTC offsets. The model calculates dates such as "Monday" or
			// "tomorrow" incorrectly, and it does not know when daylight saving time
			// starts or ends, so it reads them from this list. The offset is the one
			// at noon, because a clock change is at night.
			var days []string
			for i := range 7 {
				d := t.AddDate(0, 0, i)
				noon := time.Date(d.Year(), d.Month(), d.Day(), 12, 0, 0, 0, d.Location())
				days = append(days, noon.Format("Monday 2006-01-02, UTC offset -07:00"))
			}
			days[0] += " (today)"
			days[1] += " (tomorrow)"

			// Finally, return the time as JSON, with the weekday so that the model
			// does not have to calculate it.
			return toJSON(struct {
				Time     string   `json:"time"`
				Timezone string   `json:"timezone"`
				Weekday  string   `json:"weekday"`
				NextDays []string `json:"next_days"`
			}{
				Time:     t.Format(time.RFC3339),
				Timezone: t.Location().String(),
				Weekday:  t.Weekday().String(),
				NextDays: days,
			})
		},
	}
}
