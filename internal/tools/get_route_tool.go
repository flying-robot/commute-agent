package tools

import (
	"bytes"
	"cmp"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"maps"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// tomtomURL is the base URL of the TomTom Search and Routing APIs, and osmTileURL
// is the base URL of the OpenStreetMap tile server.
const (
	tomtomURL  = "https://api.tomtom.com"
	osmTileURL = "https://tile.openstreetmap.org/%d/%d/%d.png"
)

// congestedThreshold is the minimum traffic delay in seconds to consider a route congested.
const congestedThreshold = 60 * 15 // 15 minutes in seconds

// nowTolerance is how far in the past a departure can be and still mean "now".
// The model often gives the current time as the departure, and that time is a
// few seconds in the past when the tool gets it.
const nowTolerance = 5 * time.Minute

// maxAlternatives is the number of other routes that TomTom calculates in
// addition to the best one. TomTom accepts up to 5.
const maxAlternatives = 3

// mapWidth and mapHeight are the size of a route map in pixels.
const mapWidth, mapHeight = 640, 400

// maxAge is how long the tools keep a stored route, route map, or geocoding
// result, so that the stores do not grow without limit while the server runs.
// A map link in an old answer stops working after this time.
const maxAge = 24 * time.Hour

// indefiniteDelay is the magnitudeOfDelay that TomTom gives a road closure or a
// different incident with an indefinite delay. Such an incident can have a
// delay of 0, but it is important for a trip.
const indefiniteDelay = 4

// avoidOptions are the kinds of road that a route can avoid.
var avoidOptions = []string{"tollRoads", "motorways", "ferries", "unpavedRoads"}

// routeMaps has the points of each route that has a map, by the ID in its link.
// The get_route tool adds the routes, and the RouteMaps handler draws them. It
// is shared at the package level, so that the get_route tool is created in the
// same way as the other tools.
var routeMaps = struct {
	sync.Mutex
	points map[string]stamped[[]any]
}{points: map[string]stamped[[]any]{}}

// places has the address and the "lat,lon" position of each geocoded place, by
// its query in lower case, because a place does not move. The get_route and
// get_route_weather tools share it, so that the weather of a route needs no new
// geocoding requests.
var places = struct {
	sync.Mutex
	byQuery map[string]stamped[[2]string]
}{byQuery: map[string]stamped[[2]string]{}}

// A stamped value has the time that it was stored, so that old values can be
// deleted.
type stamped[T any] struct {
	value T
	at    time.Time
}

// GetRoute returns a tool that gives step-by-step driving directions between two
// places, with live traffic. The start and the destination can be any plain text
// address or place name, because the tool geocodes them with TomTom first.
//
// The departure can be a time in the future. Then TomTom uses the usual traffic
// for that day and hour, and avoids the road closures that are planned for that
// time, such as roadworks or the closures for an event. The tool can also take
// an arrival time instead, and then gives the time to leave.
//
// The route can avoid kinds of road, such as toll roads, so that it follows the
// preferences of the user.
//
// Each traffic incident that delays the route has its cause, its road, and
// where it starts and ends, from the TomTom Traffic API.
//
// The tool also gives a short summary of the other routes, so that the user can
// choose a different one. The model then calls the tool again with the option
// of that route to get its steps.
//
// Each result has a link to a map of the chosen route, which the RouteMaps
// handler serves.
//
// Each result also has news_about: the towns of the start and the destination
// and the main roads of the route, which the conditions agent gives to the
// get_current_news tool. It has no street addresses, because they can be the
// home of the user, and the news tool sends them to WTOP.
func GetRoute(apiKey string) internal.Tool {
	// TomTom calculates the other routes again on each request, and they change.
	// So the tool keeps the routes of the last request for each start and
	// destination, time, and roads to avoid, and uses them for an option from 2
	// up. Then the steps are for the route that the user chose from the list.
	// Option 1 always gets new routes with live traffic.
	var mu sync.Mutex
	last := map[string]stamped[[]any]{}

	return internal.Tool{
		Name:        "get_route",
		Description: "Get step-by-step driving directions between two places with live traffic, and find out if the route is congested or slower than usual. The result lists the incidents that delay the route, with their cause and road. The result also lists other routes, from the fastest to the slowest, with the main roads of each. If the user does not want a route, offer the next one in the list, and call this tool again with its option to get its steps. For a trip at a later time, give the departure: the route then avoids the road closures that are planned for that time. To find when to leave, give the arrival instead: the departure in the result is the time to leave.",
		Parameters: []internal.Parameter{{
			Name:        "start",
			Type:        "string",
			Description: "The start, as an address or a place name. Add the state or country if the name is ambiguous.",
			Required:    true,
		}, {
			Name:        "destination",
			Type:        "string",
			Description: "The destination, as an address or a place name. Add the state or country if the name is ambiguous.",
			Required:    true,
		}, {
			Name:        "option",
			Type:        "integer",
			Description: "The option of the route to give steps for, from the list of other routes. The default is 1, the fastest route.",
		}, {
			Name:        "departure",
			Type:        "string",
			Description: "The departure time in the future, in RFC 3339 format with the UTC offset of the start, for example 2026-10-19T08:00:00-04:00. Use the get_current_time tool to find the date. Leave it out to leave now.",
		}, {
			Name:        "arrival",
			Type:        "string",
			Description: "The time to arrive by, in the same format as the departure, when the user asks when to leave. Do not give both a departure and an arrival.",
		}, {
			Name:        "avoid",
			Type:        "string",
			Description: "Kinds of road to avoid, separated by commas, from: " + strings.Join(avoidOptions, ", ") + ". Use the saved preferences of the user.",
		}},
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments, and make sure that the model gave both places.
			in, err := decode(args)
			if err != nil {
				return "", err
			}
			start, destination := get[string](in, "start"), get[string](in, "destination")
			if start == "" || destination == "" {
				return "", errors.New("start and destination are required")
			}
			// The option is optional, so a missing or invalid value becomes option 1,
			// the fastest route.
			option := max(int(get[float64](in, "option")), 1)

			// The departure and the arrival are optional too, and the route leaves
			// now if both are missing. A time in the past is an error, because
			// TomTom only calculates routes from now on, but a departure a few
			// minutes in the past means now.
			when := url.Values{"departAt": {"now"}}
			departure, arrival := get[string](in, "departure"), get[string](in, "arrival")
			switch {
			case departure != "" && arrival != "":
				return "", errors.New("give a departure or an arrival, not both")
			case departure != "":
				t, err := parseTime("departure", departure)
				if err != nil {
					return "", err
				}
				if t.After(time.Now()) {
					when.Set("departAt", t.Format(time.RFC3339))
				} else if time.Since(t) > nowTolerance {
					return "", fmt.Errorf("departure %q is in the past, leave it out to leave now", departure)
				}
			case arrival != "":
				t, err := parseTime("arrival", arrival)
				if err != nil {
					return "", err
				}
				if t.Before(time.Now()) {
					return "", fmt.Errorf("arrival %q is in the past", arrival)
				}
				when = url.Values{"arriveAt": {t.Format(time.RFC3339)}}
			}

			// The roads to avoid go into the request as they are, so only the known
			// kinds of road are accepted.
			for a := range strings.SplitSeq(get[string](in, "avoid"), ",") {
				if a = strings.TrimSpace(a); a == "" {
					continue
				}
				if !slices.Contains(avoidOptions, a) {
					return "", fmt.Errorf("avoid %q is not one of %s", a, strings.Join(avoidOptions, ", "))
				}
				when.Add("avoid", a)
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

			// Next, use the routes of the last request for an option from 2 up,
			// if there are any. A request is necessary otherwise.
			// The positions are the key, and not the text from the model, because
			// the model can name the same place in different ways. The time and
			// the roads to avoid are part of the key, because they change the routes.
			trip := fromPos + ":" + toPos + "?" + when.Encode()
			mu.Lock()
			prev, ok := last[trip]
			mu.Unlock()
			routes := prev.value
			if option == 1 || !ok {
				if routes, err = calculateRoutes(ctx, apiKey, fromPos, toPos, when); err != nil {
					return "", fmt.Errorf("failed to calculate route from %q to %q: %w", from, to, err)
				}
				mu.Lock()
				prune(last, maxAge)
				last[trip] = stamped[[]any]{routes, time.Now()}
				mu.Unlock()
			}
			// Make sure that the option exists, because the model can ask for
			// an option that is not in the list.
			if option > len(routes) {
				return "", fmt.Errorf("there are only %d routes from %q to %q", len(routes), from, to)
			}

			// Next, read the travel times of the chosen route. The minutes and the
			// flags are calculated here, so that the model does not have to do
			// arithmetic.
			r := routes[option-1]
			s := get[map[string]any](r, "summary")
			travel := get[float64](s, "travelTimeInSeconds")
			usual := get[float64](s, "historicTrafficTravelTimeInSeconds")
			delay := get[float64](s, "trafficDelayInSeconds")

			// Next, collect the traffic incidents that cause a delay, and the road
			// closures. TomTom also reports other incidents with no delay, which
			// are only noise for the model.
			var delayed []any
			for _, sec := range get[[]any](r, "sections") {
				if get[float64](sec, "delayInSeconds") > 0 || get[float64](sec, "magnitudeOfDelay") == indefiniteDelay {
					delayed = append(delayed, sec)
				}
			}

			// Next, add the details of each incident, such as its cause and its
			// road, so that the user knows why the route is slow. The details need
			// a second request, so it is sent only if there are incidents. If the
			// request fails, an incident has only its category and its delay.
			var details map[string]any
			if len(delayed) > 0 {
				details = incidentDetails(ctx, apiKey, get[[]any](r, "legs", 0, "points"))
			}
			incidents := []map[string]any{}
			for _, sec := range delayed {
				incident := map[string]any{
					"category":      get[string](sec, "simpleCategory"),
					"delay_minutes": minutes(get[float64](sec, "delayInSeconds")),
				}
				if p, ok := details[get[string](sec, "eventId")]; ok {
					var events, roads []string
					for _, e := range get[[]any](p, "events") {
						events = append(events, get[string](e, "description"))
					}
					for _, n := range get[[]any](p, "roadNumbers") {
						roads = append(roads, get[string](n))
					}
					incident["description"] = strings.Join(events, ", ")
					incident["road"] = strings.Join(roads, ", ")
					incident["from"] = get[string](p, "from")
					incident["to"] = get[string](p, "to")
				}
				incidents = append(incidents, incident)
			}

			// Next, collect the text of each step of the chosen route, with the
			// miles to the next step. The distance of a step is the difference
			// between its offset and the offset of the next step, so the last step,
			// the arrival, has no distance.
			ins := get[[]any](r, "guidance", "instructions")
			steps := []string{}
			for i, step := range ins {
				msg := get[string](step, "message")
				if i+1 < len(ins) {
					meters := get[float64](ins[i+1], "routeOffsetInMeters") - get[float64](step, "routeOffsetInMeters")
					msg = fmt.Sprintf("%s (%.1f mi)", msg, miles(meters))
				}
				steps = append(steps, msg)
			}

			// Next, summarize the other routes, so that the user can choose one by
			// its roads and its travel time without the steps of each route.
			others := []map[string]any{}
			for i, other := range routes {
				if i != option-1 {
					sum := get[map[string]any](other, "summary")
					others = append(others, map[string]any{
						"option":                i + 1,
						"via":                   via(other),
						"distance_km":           km(get[float64](sum, "lengthInMeters")),
						"travel_minutes":        minutes(get[float64](sum, "travelTimeInSeconds")),
						"traffic_delay_minutes": minutes(get[float64](sum, "trafficDelayInSeconds")),
						// Without this field, the model can think that the steps
						// of the chosen route are the steps of this route.
						"steps": fmt.Sprintf("not included, call the get_route tool with option %d", i+1),
					})
				}
			}

			// Next, keep the points of the chosen route for its map. The ID is
			// random and short, so that the model can copy the link without errors
			// and nobody can guess the link of a different route.
			id := rand.Text()[:10]
			routeMaps.Lock()
			prune(routeMaps.points, maxAge)
			routeMaps.points[id] = stamped[[]any]{get[[]any](r, "legs", 0, "points"), time.Now()}
			routeMaps.Unlock()

			// Next, find the towns and the roads of the trip for the news.
			var about []string
			for _, a := range append([]string{town(from), town(to)}, via(r)...) {
				if a != "" && !slices.Contains(about, a) {
					about = append(about, a)
				}
			}

			// Finally, return the chosen route in full and the summary of the other
			// routes as JSON.
			return toJSON(map[string]any{
				"option":                option,
				"start":                 from,
				"destination":           to,
				"via":                   via(r),
				"distance_km":           km(get[float64](s, "lengthInMeters")),
				"departure":             get[string](s, "departureTime"),
				"arrival":               get[string](s, "arrivalTime"),
				"travel_minutes":        minutes(travel),
				"no_traffic_minutes":    minutes(get[float64](s, "noTrafficTravelTimeInSeconds")),
				"usual_minutes":         minutes(usual),
				"traffic_delay_minutes": minutes(delay),
				"congested":             delay > congestedThreshold,
				"slower_than_usual":     minutes(travel) > minutes(usual),
				"incidents":             incidents,
				"steps":                 steps,
				"other_routes":          others,
				"map":                   "/maps/" + id + ".png",
				"news_about":            strings.Join(about, "; "),
			})
		},
	}
}

// town returns the town and the state of a TomTom address, for example
// "Alexandria, VA" from "3401 Richmond Hwy, Alexandria, VA 22305". It returns an
// empty string if the address has no town.
func town(address string) string {
	parts := strings.Split(address, ", ")
	if len(parts) < 2 {
		return ""
	}
	state, _, _ := strings.Cut(parts[len(parts)-1], " ")
	return parts[len(parts)-2] + ", " + state
}

// RouteMaps returns a handler that serves the map of each route that the
// get_route tool found, as a PNG image. It expects the path without its prefix,
// so the path is the ID of the map with the .png extension. A map is drawn only
// when it is asked for, so that a caller that does not show it does not cost
// any tile requests.
func RouteMaps() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		routeMaps.Lock()
		points, ok := routeMaps.points[strings.TrimSuffix(req.URL.Path, ".png")]
		routeMaps.Unlock()
		if !ok {
			http.NotFound(w, req)
			return
		}
		img, err := drawMap(req.Context(), points.value)
		if err != nil {
			http.Error(w, "failed to draw map: "+err.Error(), http.StatusBadGateway)
			return
		}
		// The map of an ID never changes, so the browser can keep it.
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Cache-Control", "private, max-age=86400")
		w.Write(img)
	})
}

// drawMap returns a PNG map of a route. It puts the base map together from
// OpenStreetMap tiles, and draws the route and its start and end on it.
func drawMap(ctx context.Context, points []any) ([]byte, error) {
	// First, project each point with Web Mercator, which the base map uses, to
	// a world that is a square of size 1. Then find the box around the route.
	if len(points) == 0 {
		return nil, errors.New("the route has no points")
	}
	pts := make([][2]float64, len(points))
	minX, minY, maxX, maxY := 1.0, 1.0, 0.0, 0.0
	for i, p := range points {
		lat := get[float64](p, "latitude") * math.Pi / 180
		x := (get[float64](p, "longitude") + 180) / 360
		y := (1 - math.Log(math.Tan(lat)+1/math.Cos(lat))/math.Pi) / 2
		pts[i] = [2]float64{x, y}
		minX, minY, maxX, maxY = min(minX, x), min(minY, y), max(maxX, x), max(maxY, y)
	}

	// Next, choose the largest zoom at which the route fits in the map with a
	// margin. At zoom z, the world is 2^z tiles of 256 pixels wide.
	zoom := 0
	for z := 18; z > 0; z-- {
		scale := 256 * math.Exp2(float64(z))
		if (maxX-minX)*scale <= mapWidth*0.85 && (maxY-minY)*scale <= mapHeight*0.85 {
			zoom = z
			break
		}
	}
	scale := 256 * math.Exp2(float64(zoom))

	// Next, find the world pixel at the top left of the map, so that the center
	// of the box is at the center of the map.
	left := (minX+maxX)/2*scale - mapWidth/2
	top := (minY+maxY)/2*scale - mapHeight/2

	// Next, draw each tile that the map covers at its place in the map. A tile
	// number wraps around the world from east to west, and there are no tiles
	// north or south of the world.
	img := image.NewRGBA(image.Rect(0, 0, mapWidth, mapHeight))
	tiles := 1 << zoom
	for ty := int(math.Floor(top / 256)); ty <= int(math.Floor((top+mapHeight)/256)); ty++ {
		for tx := int(math.Floor(left / 256)); tx <= int(math.Floor((left+mapWidth)/256)); tx++ {
			if ty < 0 || ty >= tiles {
				continue
			}
			tile, err := fetchTile(ctx, fmt.Sprintf(osmTileURL, zoom, (tx%tiles+tiles)%tiles, ty))
			if err != nil {
				return nil, err
			}
			at := image.Pt(tx*256-int(math.Round(left)), ty*256-int(math.Round(top)))
			draw.Draw(img, tile.Bounds().Add(at), tile, image.Point{}, draw.Src)
		}
	}

	// Next, draw the route as a line through the points, in pixels from the
	// top left of the map.
	pixel := func(p [2]float64) (float64, float64) {
		return p[0]*scale - math.Round(left), p[1]*scale - math.Round(top)
	}
	route := color.RGBA{0x25, 0x63, 0xeb, 0xff}
	for i := 1; i < len(pts); i++ {
		x0, y0 := pixel(pts[i-1])
		x1, y1 := pixel(pts[i])
		line(img, x0, y0, x1, y1, 3, route)
	}

	// Finally, mark the start in green and the end in red, each with a white
	// edge so that it shows on the line, and encode the map as PNG.
	for _, end := range []struct {
		p [2]float64
		c color.RGBA
	}{{pts[0], color.RGBA{0x16, 0xa3, 0x4a, 0xff}}, {pts[len(pts)-1], color.RGBA{0xdc, 0x26, 0x26, 0xff}}} {
		x, y := pixel(end.p)
		dot(img, x, y, 9, color.White)
		dot(img, x, y, 7, end.c)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// fetchTile gets one OpenStreetMap tile.
func fetchTile(ctx context.Context, tileURL string) (image.Image, error) {
	b, err := fetch(ctx, tileURL, "")
	if err != nil {
		return nil, fmt.Errorf("failed to get tile: %w", err)
	}
	return png.Decode(bytes.NewReader(b))
}

// line draws a line of radius r from one point to another. It draws a dot at
// each pixel along the line, which is simple and fast enough for a route.
func line(img *image.RGBA, x0, y0, x1, y1, r float64, c color.Color) {
	n := max(math.Abs(x1-x0), math.Abs(y1-y0), 1)
	for i := 0.0; i <= n; i++ {
		dot(img, x0+(x1-x0)*i/n, y0+(y1-y0)*i/n, r, c)
	}
}

// dot draws a filled circle of radius r. Pixels outside the image are ignored.
func dot(img *image.RGBA, x, y, r float64, c color.Color) {
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			if dx*dx+dy*dy <= r*r {
				img.Set(int(x+dx), int(y+dy), c)
			}
		}
	}
}

// calculateRoutes asks TomTom for the routes between two positions with traffic
// at the departure or arrival time, and returns them from the fastest to the slowest. The travel times for
// "all" include the time without traffic and the usual time on this day and
// hour, so that the delays can be compared.
func calculateRoutes(ctx context.Context, apiKey, fromPos, toPos string, when url.Values) ([]any, error) {
	// First, request the best route and the alternatives, with the steps as text
	// and the traffic sections that show where the delays are. The traffic must be
	// on, because without it TomTom also ignores the planned road closures. The
	// time and the roads to avoid come from when.
	query := url.Values{
		"key":                  {apiKey},
		"traffic":              {"true"},
		"computeTravelTimeFor": {"all"},
		"sectionType":          {"traffic"},
		"instructionsType":     {"text"},
		"language":             {"en-US"},
		"maxAlternatives":      {strconv.Itoa(maxAlternatives)},
	}
	maps.Copy(query, when)
	res, err := getJSON(ctx, "/routing/1/calculateRoute/"+fromPos+":"+toPos+"/json", query)
	if err != nil {
		return nil, err
	}

	// Finally, sort the routes from the fastest to the slowest, so that the
	// option numbers go from the best route to the worst. TomTom does not sort
	// its alternatives. The sort is stable, so that TomTom's best route stays
	// first if two routes take the same time.
	routes := get[[]any](res, "routes")
	if len(routes) == 0 {
		return nil, errors.New("no route found")
	}
	slices.SortStableFunc(routes, func(a, b any) int {
		return cmp.Compare(get[float64](a, "summary", "travelTimeInSeconds"), get[float64](b, "summary", "travelTimeInSeconds"))
	})
	return routes, nil
}

// incidentDetails returns the details of the traffic incidents in the box around
// the points of a route, by the event ID that the Routing API gives each traffic
// section. The Traffic API gives each incident a longer ID that ends with "-"
// and that event ID. It returns nil if the request fails, for example if the box
// is larger than the 10,000 square kilometers that TomTom accepts, so that a
// route never fails because of the details.
func incidentDetails(ctx context.Context, apiKey string, points []any) map[string]any {
	// First, find the box around the route.
	minLat, minLon, maxLat, maxLon := 90.0, 180.0, -90.0, -180.0
	for _, p := range points {
		lat, lon := get[float64](p, "latitude"), get[float64](p, "longitude")
		minLat, minLon, maxLat, maxLon = min(minLat, lat), min(minLon, lon), max(maxLat, lat), max(maxLon, lon)
	}

	// Next, ask for the incidents in the box. Future incidents are included,
	// because a route at a later time has the planned closures in it.
	res, err := getJSON(ctx, "/traffic/services/5/incidentDetails", url.Values{
		"key":                {apiKey},
		"bbox":               {fmt.Sprintf("%f,%f,%f,%f", minLon, minLat, maxLon, maxLat)},
		"fields":             {"{incidents{properties{id,events{description},from,to,roadNumbers}}}"},
		"language":           {"en-US"},
		"timeValidityFilter": {"present,future"},
	})
	if err != nil {
		return nil
	}

	// Finally, key the incidents by the event ID at the end of their ID.
	details := map[string]any{}
	for _, inc := range get[[]any](res, "incidents") {
		id := get[string](inc, "properties", "id")
		details[id[strings.LastIndex(id, "-")+1:]] = get[map[string]any](inc, "properties")
	}
	return details
}

// via returns the two roads on which a route goes the longest distance, so that
// the user can tell the routes apart. A road number such as "I-95 S" is used
// without its direction, and a street name is used when a road has no number.
func via(route any) []string {
	// First, add up the distance on each road. Each step names the road that
	// comes after it, and the road continues until the offset of the next step.
	ins := get[[]any](route, "guidance", "instructions")
	dist := map[string]float64{}
	for i := 0; i+1 < len(ins); i++ {
		road := get[string](ins[i], "roadNumbers", 0)
		for _, d := range []string{" N", " S", " E", " W"} {
			road = strings.TrimSuffix(road, d)
		}
		if road == "" {
			road = get[string](ins[i], "street")
		}
		if road != "" {
			dist[road] += get[float64](ins[i+1], "routeOffsetInMeters") - get[float64](ins[i], "routeOffsetInMeters")
		}
	}

	// Finally, sort the roads from the longest distance to the shortest, and
	// return the first two. The roads are sorted by name first, so that two roads
	// with the same distance are always in the same order.
	roads := slices.Sorted(maps.Keys(dist))
	slices.SortStableFunc(roads, func(a, b string) int { return cmp.Compare(dist[b], dist[a]) })
	return roads[:min(2, len(roads))]
}

// geocode returns the address and the "lat,lon" position of the best match that
// TomTom finds for a plain text query. It uses fuzzy search and not the geocode
// endpoint, because fuzzy search also finds places of interest, such as
// "Washington Monument".
//
// Each result is kept in places, so that the same query needs no new request.
func geocode(ctx context.Context, apiKey, query string) (string, string, error) {
	// First, use the stored result of the query if there is one. The case of
	// the query does not change the place.
	key := strings.ToLower(strings.TrimSpace(query))
	places.Lock()
	p, ok := places.byQuery[key]
	places.Unlock()
	if ok {
		return p.value[0], p.value[1], nil
	}

	// Next, search for the query, and keep only the best match.
	res, err := getJSON(ctx, "/search/2/search/"+url.PathEscape(query)+".json", url.Values{"key": {apiKey}, "limit": {"1"}})
	if err != nil {
		return "", "", fmt.Errorf("failed to find %q: %w", query, err)
	}
	r := get[map[string]any](res, "results", 0)
	if r == nil {
		return "", "", fmt.Errorf("no location found for %q", query)
	}

	// Next, add the name of a place of interest to its address, so that the
	// model and the user can see which place TomTom found.
	address := get[string](r, "address", "freeformAddress")
	if name := get[string](r, "poi", "name"); name != "" {
		address = name + ", " + address
	}

	// Finally, keep and return the position in the "lat,lon" form that the
	// Routing API uses in its path. The result is also kept by its address,
	// because the model gives the address from a route result to the
	// get_route_weather tool.
	pos := fmt.Sprintf("%f,%f", get[float64](r, "position", "lat"), get[float64](r, "position", "lon"))
	result := stamped[[2]string]{[2]string{address, pos}, time.Now()}
	places.Lock()
	prune(places.byQuery, maxAge)
	places.byQuery[key] = result
	places.byQuery[strings.ToLower(address)] = result
	places.Unlock()
	return address, pos, nil
}

// prune deletes the values in m that were stored more than age ago. The caller
// must hold the lock of m.
func prune[K comparable, T any](m map[K]stamped[T], age time.Duration) {
	maps.DeleteFunc(m, func(_ K, s stamped[T]) bool { return time.Since(s.at) > age })
}

// getJSON sends a GET request to the TomTom API and returns the decoded JSON
// body. An error never has the URL in it, because the URL has the API key.
func getJSON(ctx context.Context, path string, query url.Values) (any, error) {
	// First, create the request. The context cancels the request if the caller
	// of the agent disconnects.
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, tomtomURL+path+"?"+query.Encode(), nil)
	if err != nil {
		return nil, errors.New("failed to create request")
	}

	// Next, send the request. An error from the client is a *url.Error, which
	// has the URL and the API key in it, so return only its cause.
	resp, err := do(req)
	if err != nil {
		if uerr, ok := errors.AsType[*url.Error](err); ok {
			err = uerr.Err
		}
		return nil, err
	}
	defer resp.Body.Close()

	// Next, read and decode the body. An error body is JSON too, so the body is
	// decoded before the status is checked.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var out any
	jsonErr := json.Unmarshal(body, &out)

	// Finally, return the error or the body. TomTom puts the reason for a failure
	// in detailedError.message, for example NO_ROUTE_FOUND. If the body does not
	// have it, for example an HTML page from a proxy, use the whole body.
	if resp.StatusCode != http.StatusOK {
		if msg := get[string](out, "detailedError", "message"); msg != "" {
			return nil, fmt.Errorf("status %d: %s", resp.StatusCode, msg)
		}
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}
	return out, jsonErr
}

// get returns the value at a path of object keys and array indexes in decoded
// JSON. If the path does not exist, or the value is not a T, it returns the zero
// value of T. JSON numbers are float64.
func get[T any](v any, path ...any) T {
	// Follow each part of the path. A string is a key of an object, and an int
	// is an index of an array. A part that does not match makes the value nil.
	for _, p := range path {
		switch k := p.(type) {
		case string:
			m, _ := v.(map[string]any)
			v = m[k]
		case int:
			a, _ := v.([]any)
			if k >= len(a) {
				v = nil
			} else {
				v = a[k]
			}
		}
	}
	t, _ := v.(T)
	return t
}

// parseTime parses a time from the model in RFC 3339 format. The name of the
// argument is in the error, so that the model knows which one to correct.
func parseTime(name, value string) (time.Time, error) {
	t, err := time.Parse(time.RFC3339, value)
	if err != nil {
		return time.Time{}, fmt.Errorf("%s %q is not in RFC 3339 format with a UTC offset", name, value)
	}
	return t, nil
}

// minutes converts seconds to whole minutes, rounded to the nearest minute.
func minutes(seconds float64) int {
	return int(math.Round(seconds / 60))
}

// km converts meters to kilometers, rounded to one decimal place.
func km(meters float64) float64 {
	return math.Round(meters/100) / 10
}

// miles converts meters to miles, rounded to one decimal place.
func miles(meters float64) float64 {
	return math.Round(meters/1609.344*10) / 10
}
