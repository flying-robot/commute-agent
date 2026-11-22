package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// weatherURL is the base URL of the National Weather Service API. It is free,
// needs no key, and covers only the United States.
const weatherURL = "https://api.weather.gov"

// weatherCollection is the collection of the vector store that has the weather
// of each place.
const weatherCollection = "weather"

// placeWeather is the weather of one place for the model, and the time that it
// was checked if it is a saved one.
type placeWeather struct {
	result  map[string]any
	checked time.Time
	err     error
}

// GetRouteWeather returns a tool that gives the weather of a trip: the forecast
// at the start when the trip leaves, and at the destination when it arrives,
// with the active weather alerts at each place. The weather comes from the
// National Weather Service, so it is only for places in the United States. The
// tool geocodes the places with TomTom, in the same way as the get_route tool.
//
// The tool saves the weather of each place in the vector store for ttl, so that
// the model can find it again with the search_saved_data tool. If the weather of
// a place for the same hour was checked in that time, the tool gives the saved
// weather, and tells the model so, unless the user asked to check again. After
// ttl, the model must get the weather again, because alerts change quickly.
func GetRouteWeather(apiKey string, save internal.SaveFunc, search internal.SearchFunc, ttl time.Duration) internal.Tool {
	return internal.Tool{
		Name:        "get_route_weather",
		Description: "Get the weather forecast and the weather alerts for a trip: at the start when the trip leaves, and at the destination when it arrives. Give the start, the destination, the departure, and the arrival exactly as the get_route result gives them. The weather of a place is null if there is no forecast for it, for example outside the United States or more than about six days ahead.",
		Parameters: []internal.Parameter{{
			Name:        "start",
			Type:        "string",
			Description: "The start of the trip, as an address or a place name.",
			Required:    true,
		}, {
			Name:        "destination",
			Type:        "string",
			Description: "The destination of the trip, as an address or a place name.",
			Required:    true,
		}, {
			Name:        "departure",
			Type:        "string",
			Description: "The departure time of the trip, in RFC 3339 format with a UTC offset, for example 2026-10-19T08:00:00-04:00.",
			Required:    true,
		}, {
			Name:        "arrival",
			Type:        "string",
			Description: "The arrival time of the trip, in the same format as the departure.",
			Required:    true,
		}, {
			Name:        "force",
			Type:        "boolean",
			Description: "Get the weather again even if it was checked a short time ago. Set it to true only when the user asks you to check the weather again.",
		}},
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments, and make sure that the model gave both
			// places and both times.
			in, err := decode(args)
			if err != nil {
				return "", err
			}
			start, destination := get[string](in, "start"), get[string](in, "destination")
			if start == "" || destination == "" {
				return "", errors.New("start and destination are required")
			}
			leave, err := parseTime("departure", get[string](in, "departure"))
			if err != nil {
				return "", err
			}
			arrive, err := parseTime("arrival", get[string](in, "arrival"))
			if err != nil {
				return "", err
			}
			if apiKey == "" {
				return "", errors.New("TOMTOM_API_KEY is not set")
			}

			// Next, find the coordinates of each place.
			from, fromPos, err := geocode(ctx, apiKey, start)
			if err != nil {
				return "", err
			}
			to, toPos, err := geocode(ctx, apiKey, destination)
			if err != nil {
				return "", err
			}

			// Next, get the weather at the start when the trip leaves, and at the
			// destination when it arrives. The two places are independent, so they
			// are done at the same time.
			force := get[bool](in, "force")
			var atStart, atEnd placeWeather
			var wg sync.WaitGroup
			wg.Go(func() { atStart = weatherFor(ctx, save, search, ttl, from, fromPos, leave, force) })
			wg.Go(func() { atEnd = weatherFor(ctx, save, search, ttl, to, toPos, arrive, force) })
			wg.Wait()
			if err := errors.Join(atStart.err, atEnd.err); err != nil {
				return "", err
			}

			// Finally, return the weather of each place as JSON, with the places that
			// TomTom found, so that the model can make sure that they are the places
			// of the route. If the weather of a place is a saved one, the result
			// says how old the oldest one is, so that the caller can tell the user.
			out := map[string]any{"start": atStart.result, "destination": atEnd.result}
			var saved []time.Time
			for _, p := range []placeWeather{atStart, atEnd} {
				if !p.checked.IsZero() {
					saved = append(saved, p.checked)
				}
			}
			if len(saved) > 0 {
				out[internal.SavedMinutesAgo] = int(time.Since(slices.MinFunc(saved, time.Time.Compare)).Minutes())
			}
			return toJSON(out)
		},
	}
}

// weatherFor returns the weather of a place at a time for the model. It uses the
// saved weather of the position for the hour of t, if there is one and force is
// false, because the forecast is by the hour. Otherwise it gets the weather and
// saves it. The saved text is the JSON of the result, so that it can be read
// back, and the model can read it in a search of the saved data.
func weatherFor(ctx context.Context, save internal.SaveFunc, search internal.SearchFunc, ttl time.Duration, place, pos string, t time.Time, force bool) placeWeather {
	// First, use the saved weather, if there is one.
	source := pos + "@" + t.Truncate(time.Hour).Format(time.RFC3339)
	if !force {
		docs, err := search(ctx, weatherCollection, source, "weather at "+place, 1)
		if err != nil {
			return placeWeather{err: fmt.Errorf("failed to search the saved weather: %w", err)}
		}
		if len(docs) > 0 {
			var result map[string]any
			if err := json.Unmarshal([]byte(docs[0].Text), &result); err != nil {
				return placeWeather{err: fmt.Errorf("failed to read the saved weather: %w", err)}
			}
			return placeWeather{result: result, checked: docs[0].SavedAt}
		}
	}

	// Finally, get the weather, and save it for ttl.
	result := map[string]any{"place": place, "time": t.Format(time.RFC3339), "weather": weatherAt(ctx, pos, t)}
	b, err := json.Marshal(result)
	if err != nil {
		return placeWeather{err: err}
	}
	doc := internal.Document{ID: source, Source: source, Text: string(b)}
	if err := save(ctx, weatherCollection, []internal.Document{doc}, ttl); err != nil {
		return placeWeather{err: fmt.Errorf("failed to save the weather: %w", err)}
	}
	return placeWeather{result: result}
}

// weatherAt returns the forecast for the hour of t at a "lat,lon" position, and
// the weather alerts that are active there now. It returns nil if there is no
// forecast for the place or the time, for example outside the United States or
// more than about six days ahead, so that a route never fails because of the
// weather.
func weatherAt(ctx context.Context, pos string, t time.Time) map[string]any {
	// First, find the hourly forecast of the place. The National Weather Service
	// gives each place a forecast URL for its grid square.
	point, err := getWeatherJSON(ctx, weatherURL+"/points/"+pos)
	if err != nil {
		return nil
	}
	hourly, err := getWeatherJSON(ctx, get[string](point, "properties", "forecastHourly"))
	if err != nil {
		return nil
	}

	// Next, find the hour that has the time in it.
	var period map[string]any
	for _, p := range get[[]any](hourly, "properties", "periods") {
		start, err1 := time.Parse(time.RFC3339, get[string](p, "startTime"))
		end, err2 := time.Parse(time.RFC3339, get[string](p, "endTime"))
		if err1 == nil && err2 == nil && !t.Before(start) && t.Before(end) {
			period, _ = p.(map[string]any)
			break
		}
	}
	if period == nil {
		return nil
	}

	// Next, get the alerts, such as a winter storm warning. An alert is
	// important for a trip even if it is not for the hour of the trip. Only
	// actual alerts are asked for, because the service also sends test messages.
	alerts := []string{}
	if res, err := getWeatherJSON(ctx, weatherURL+"/alerts/active?status=actual&point="+pos); err == nil {
		// Some alerts have no headline, so their event, such as "Frost
		// Advisory", is used instead.
		for _, f := range get[[]any](res, "features") {
			alert := get[string](f, "properties", "headline")
			if alert == "" {
				alert = get[string](f, "properties", "event")
			}
			if alert != "" {
				alerts = append(alerts, alert)
			}
		}
	}

	// Finally, return the forecast in the units that the forecast uses.
	return map[string]any{
		"forecast":             get[string](period, "shortForecast"),
		"temperature_f":        get[float64](period, "temperature"),
		"precipitation_chance": get[float64](period, "probabilityOfPrecipitation", "value"),
		"wind":                 get[string](period, "windSpeed") + " " + get[string](period, "windDirection"),
		"alerts":               alerts,
	}
}

// getWeatherJSON sends a GET request to the National Weather Service API and
// returns the decoded JSON body.
func getWeatherJSON(ctx context.Context, weatherURL string) (any, error) {
	if weatherURL == "" {
		return nil, errors.New("no URL")
	}
	b, err := fetch(ctx, weatherURL, "application/geo+json")
	if err != nil {
		return nil, err
	}
	var out any
	return out, json.Unmarshal(b, &out)
}
