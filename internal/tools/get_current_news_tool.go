package tools

import (
	"cmp"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"maps"
	"math"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// newsFeedURL is the RSS feed of the latest WTOP news for one area, and
// newsSearchURL is the RSS feed of a WTOP search.
const (
	newsFeedURL   = "https://wtop.com/%s/feed/"
	newsSearchURL = "https://wtop.com/feed/?s=%s"
)

// newsCollection is the collection of the vector store that has the articles.
const newsCollection = "news"

// minNewsResults and maxNewsResults limit the number of articles that the tool
// gives for a trip. A small model finds the important events more reliably in a
// few relevant articles than in a full feed.
const (
	minNewsResults = 5
	maxNewsResults = 10
)

// newsMinScore is the smallest similarity that an article must have to the trip
// to be given. With nomic-embed-text, articles about a place or a road of the
// trip score about 0.68 to 0.73, and unrelated articles about 0.55 to 0.62, so
// this limit removes the unrelated ones. A different embedding model needs a
// different limit.
const newsMinScore = 0.64

// titleEscaper escapes the characters that Markdown uses for links, images,
// emphasis, code, and HTML, and linkEscaper encodes the characters that end the
// link of a Markdown link. Then the title and the link of an article cannot
// change the format of the list or add a link.
var (
	titleEscaper = strings.NewReplacer(`\`, `\\`, "[", `\[`, "]", `\]`, "(", `\(`, ")", `\)`, "*", `\*`, "_", `\_`, "`", "\\`", "<", `\<`, ">", `\>`, "!", `\!`, "\n", " ")
	linkEscaper  = strings.NewReplacer("(", "%28", ")", "%29", " ", "%20", "<", "%3C", ">", "%3E", "\n", "")
)

// newsAreas are the areas that WTOP has a local feed for.
var newsAreas = []string{"dc", "maryland", "virginia"}

// feedItem is the part of an RSS item that the tool uses. The XML decoder needs
// a struct, because it cannot decode into a map.
type feedItem struct {
	Title       string   `xml:"title"`
	Link        string   `xml:"link"`
	PubDate     string   `xml:"pubDate"`
	Description string   `xml:"description"`
	Categories  []string `xml:"category"`
}

// GetCurrentNews returns a tool that gives the latest local news from WTOP,
// covering DC, Maryland, and Virginia. The tool does not decide which news is
// important. It gives the title and the date of each article to the model, so
// that the model can find events that affect a trip but are not in the route
// yet. It does not give the summary, because the text of an article can have
// instructions in it, and a shorter text gives them less space.
//
// The tool saves each article in the vector store for ttl, so that the
// model can find it again with the search_saved_data tool. If the feed was
// checked in that time, the tool gives the saved articles, and tells the model
// so, unless the user asked to check again. When the model gives the places
// and the roads of a trip, the tool gives only the articles whose meaning is
// nearest to them. With only the places and the roads, the tool searches WTOP
// for each of them, and gives the n nearest articles of all the searches. n is
// limited to minNewsResults to maxNewsResults. After ttl, the model must get the
// news again, because news changes quickly.
func GetCurrentNews(save internal.SaveFunc, search internal.SearchFunc, ttl time.Duration, n int) internal.Tool {
	n = min(max(n, minNewsResults), maxNewsResults)
	return internal.Tool{
		Name:        "get_current_news",
		Description: "Get the latest local news from WTOP for DC, Maryland, or Virginia, or search WTOP for a road, a place, or an event. Use it to find events that can affect a trip but are not in the route yet, such as races, parades, protests, crashes, or road closures. For a trip, give the news_about of the get_route result as about. If an article affects the route, use the other routes or a different departure from the get_route tool to go around it.",
		Parameters: []internal.Parameter{{
			Name:        "area",
			Type:        "string",
			Description: "The area to get the latest news for. Leave it out to search.",
			Enum:        newsAreas,
		}, {
			Name:        "query",
			Type:        "string",
			Description: "Words to search for in all WTOP news, for example Marine Corps Marathon or Beltway. Leave it out to get the latest news for an area.",
		}, {
			Name:        "about",
			Type:        "string",
			Description: "The places and the main roads of the trip, separated by semicolons, for example: Bethesda, MD; Arlington, VA; I-495; VA-267. The tool then gives only the articles that are most related to them. Without an area or a query, the tool searches WTOP for each place and road. Leave it out to get all the articles.",
		}, {
			Name:        "force",
			Type:        "boolean",
			Description: "Get the news again even if it was checked a short time ago. Set it to true only when the user asks you to check the news again.",
		}},
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			// First, decode the arguments, and choose the feeds. The area goes into
			// the path of the URL, so only the known areas are accepted. With only
			// about, the tool searches WTOP for each place and road in it, so that
			// the locations of the trip are given to WTOP.
			in, err := decode(args)
			if err != nil {
				return "", err
			}
			area, query, about := get[string](in, "area"), get[string](in, "query"), get[string](in, "about")
			force := get[bool](in, "force")
			var feedURL, from string
			switch {
			case query != "":
				feedURL, from = fmt.Sprintf(newsSearchURL, url.QueryEscape(query)), "WTOP search for "+query
			case slices.Contains(newsAreas, area):
				feedURL, from = fmt.Sprintf(newsFeedURL, area), "WTOP "+area+" news"
			case about == "":
				return "", fmt.Errorf("give a query, about, or an area that is one of %s", strings.Join(newsAreas, ", "))
			}
			if feedURL != "" {
				docs, savedAt, err := checkFeed(ctx, save, search, ttl, n, feedURL, from, about, force)
				if err != nil {
					return "", err
				}
				return newsResult(area, query, about, docs, savedAt)
			}

			// Next, search WTOP for each location of the trip, and keep each article
			// once, with its highest score. The result is saved data only if no
			// feed was checked again, and its age is then the age of the oldest
			// feed.
			best := map[string]internal.Document{}
			var savedAt time.Time
			fresh := false
			for loc := range strings.SplitSeq(about, ";") {
				if loc = strings.TrimSpace(loc); loc == "" {
					continue
				}
				docs, at, err := checkFeed(ctx, save, search, ttl, n, fmt.Sprintf(newsSearchURL, url.QueryEscape(loc)), "WTOP search for "+loc, about, force)
				if err != nil {
					return "", err
				}
				if at.IsZero() {
					fresh = true
				} else if savedAt.IsZero() || at.Before(savedAt) {
					savedAt = at
				}
				for _, d := range docs {
					if b, ok := best[d.ID]; !ok || d.Score > b.Score {
						best[d.ID] = d
					}
				}
			}
			if fresh {
				savedAt = time.Time{}
			}

			// Finally, return the n articles that are most related to the trip.
			docs := slices.SortedFunc(maps.Values(best), func(a, b internal.Document) int { return cmp.Compare(b.Score, a.Score) })
			return newsResult(area, query, about, docs[:min(n, len(docs))], savedAt)
		},
	}
}

// checkFeed returns the articles of one feed, and the time that they were saved
// if they are saved ones. If the model gave the places and the roads of a trip
// as about, it returns only the n articles that are most related to them. It
// saves new articles for ttl.
func checkFeed(ctx context.Context, save internal.SaveFunc, search internal.SearchFunc, ttl time.Duration, n int, feedURL, from, about string, force bool) ([]internal.Document, time.Time, error) {
	// First, use the saved articles of the feed if it was checked in the last
	// ttl, unless the user asked to check again. Expired articles are
	// deleted, so any saved article is new enough. The result says when the
	// check was, so that the model can tell the user. The articles are the ones
	// nearest to the trip, or all of them.
	limit, ask := math.MaxInt, from
	if about != "" {
		limit, ask = n, about
	}
	if !force {
		saved, err := search(ctx, newsCollection, feedURL, ask, limit)
		if err != nil {
			return nil, time.Time{}, fmt.Errorf("failed to search the saved news: %w", err)
		}
		if len(saved) > 0 {
			// The time is read first, because related changes saved.
			savedAt := saved[0].SavedAt
			return related(saved, about), savedAt, nil
		}
	}

	// Next, get the articles from the feed.
	items, err := fetchFeed(ctx, feedURL)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get WTOP news: %w", err)
	}

	// Next, write each article as plain text, which the model reads and the
	// vector store embeds. The age is calculated here, so that the model does
	// not have to do arithmetic with dates. A search also finds old articles.
	// The link is the ID, so that a new check of the feed replaces an article
	// and does not add a copy.
	docs := []internal.Document{}
	for _, it := range items {
		published := "unknown"
		if t, err := time.Parse(time.RFC1123Z, it.PubDate); err == nil {
			published = fmt.Sprintf("%s (%d days ago)", t.Format(time.RFC3339), int(math.Floor(time.Since(t).Hours()/24)))
		}
		docs = append(docs, internal.Document{ID: it.Link, Source: feedURL, Text: fmt.Sprintf(
			"Title: %s\nFrom: %s\nPublished: %s\nCategories: %s\nSummary: %s\nLink: %s",
			it.Title, from, published, strings.Join(it.Categories, ", "), it.Description, it.Link,
		)})
	}

	// Next, save the articles, so that the tool and the model can find them
	// later.
	if err := save(ctx, newsCollection, docs, ttl); err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to save the news: %w", err)
	}

	// Finally, if the model gave the places and the roads of a trip, keep only
	// the articles of the feed that are nearest to them.
	if about != "" {
		if docs, err = search(ctx, newsCollection, feedURL, about, n); err != nil {
			return nil, time.Time{}, fmt.Errorf("failed to search the news: %w", err)
		}
	}
	return related(docs, about), time.Time{}, nil
}

// related returns the documents whose score is at least newsMinScore, if the
// model gave the places and the roads of a trip. Otherwise it returns all of
// them, because their score is not a similarity to a trip.
func related(docs []internal.Document, about string) []internal.Document {
	if about == "" {
		return docs
	}
	return slices.DeleteFunc(docs, func(d internal.Document) bool { return d.Score < newsMinScore })
}

// newsResult returns the title and the date of each article as JSON, from the
// newest to the oldest, or from the most related to the least related. If
// savedAt is not zero, the articles are saved ones from that time, and the
// result says how old they are. The result also has a list of the titles with
// their links, which the answer must include, so that the user always sees what
// the news check found.
func newsResult(area, query, about string, docs []internal.Document, savedAt time.Time) (string, error) {
	articles := []string{}
	scope := ""
	if about != "" {
		scope = " related to this trip"
	}
	mention := "**News:** WTOP has no articles" + scope + "."
	if len(docs) > 0 {
		mention = "**News:** WTOP has these articles" + scope + ":\n"
	}
	for _, d := range docs {
		title := articleField(d.Text, "Title")
		articles = append(articles, "Title: "+title+"\nPublished: "+articleField(d.Text, "Published"))
		mention += fmt.Sprintf("\n- [%s](%s)", titleEscaper.Replace(title), linkEscaper.Replace(articleField(d.Text, "Link")))
	}
	out := map[string]any{
		"source":         "WTOP",
		"area":           area,
		"query":          query,
		"about":          about,
		"articles":       articles,
		internal.Mention: mention,
	}
	if about != "" && len(articles) == 0 {
		out["note"] = "No article of this feed is related to the trip."
	}
	if !savedAt.IsZero() {
		out[internal.SavedMinutesAgo] = int(time.Since(savedAt).Minutes())
	}
	return toJSON(out)
}

// articleField returns the value of the line "name: value" of an article text.
func articleField(text, name string) string {
	for line := range strings.Lines(text) {
		if v, ok := strings.CutPrefix(line, name+": "); ok {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

// fetchFeed gets an RSS feed and returns its items.
func fetchFeed(ctx context.Context, feedURL string) ([]feedItem, error) {
	b, err := fetch(ctx, feedURL, "")
	if err != nil {
		return nil, err
	}
	var rss struct {
		Items []feedItem `xml:"channel>item"`
	}
	if err := xml.Unmarshal(b, &rss); err != nil {
		return nil, fmt.Errorf("failed to decode feed: %w", err)
	}
	return rss.Items, nil
}
