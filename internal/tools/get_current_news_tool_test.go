package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// article returns the text of a saved article, in the form that the tool saves.
func article(title, link string) string {
	return fmt.Sprintf("Title: %s\nFrom: WTOP\nPublished: 2026-11-16T08:00:00-05:00 (0 days ago)\nCategories: Traffic\nSummary: text\nLink: %s", title, link)
}

// newsOutput is the part of a result of the news tool that the tests check.
type newsOutput struct {
	Articles []string `json:"articles"`
	Mention  string   `json:"mention"`
	Note     string   `json:"note"`
	Saved    *int     `json:"saved_minutes_ago"`
}

// decodeNews decodes a result of the news tool.
func decodeNews(t *testing.T, out string) newsOutput {
	t.Helper()
	var n newsOutput
	if err := json.Unmarshal([]byte(out), &n); err != nil {
		t.Fatalf("result %q: %v", out, err)
	}
	return n
}

// TestNewsResult checks that each article gives only its title and its date,
// that the list of the links escapes Markdown, that a trip with no related
// article has a note, and that saved articles have their age.
func TestNewsResult(t *testing.T) {
	docs := []internal.Document{{Text: article("Crash [I-495] *closes* lanes", "https://wtop.com/a (b)")}}
	out, err := newsResult("", "", "Bethesda, MD", docs, time.Now().Add(-2*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	n := decodeNews(t, out)
	if len(n.Articles) != 1 || n.Articles[0] != "Title: Crash [I-495] *closes* lanes\nPublished: 2026-11-16T08:00:00-05:00 (0 days ago)" {
		t.Errorf("articles = %q, want the title and the date only", n.Articles)
	}
	if !strings.Contains(n.Mention, `- [Crash \[I-495\] \*closes\* lanes](https://wtop.com/a%20%28b%29)`) {
		t.Errorf("mention = %q, want an escaped title and link", n.Mention)
	}
	if !strings.Contains(n.Mention, "related to this trip") || n.Saved == nil || *n.Saved != 2 {
		t.Errorf("mention = %q, saved = %v, want a trip and an age of 2", n.Mention, n.Saved)
	}

	out, _ = newsResult("", "", "Bethesda, MD", nil, time.Time{})
	if n := decodeNews(t, out); n.Note == "" || n.Saved != nil || !strings.Contains(n.Mention, "no articles") {
		t.Errorf("no articles = %+v, want a note and no age", n)
	}
	out, _ = newsResult("dc", "", "", nil, time.Time{})
	if n := decodeNews(t, out); n.Note != "" {
		t.Errorf("no articles without a trip = %+v, want no note", n)
	}
}

// TestEscapers checks that a title cannot add a link or a line, and that a
// link cannot end early.
func TestEscapers(t *testing.T) {
	if got := titleEscaper.Replace("a](evil)\n![x]`<b>_"); got != `a\]\(evil\) \!\[x\]\`+"`"+`\<b\>\_` {
		t.Errorf("title = %q", got)
	}
	if got := linkEscaper.Replace("https://x/a b)<c>\n"); got != "https://x/a%20b%29%3Cc%3E" {
		t.Errorf("link = %q", got)
	}
}

// TestArticleField checks that a field is found by its name, and that a missing
// field is empty.
func TestArticleField(t *testing.T) {
	text := article("T", "L")
	if got := articleField(text, "Link"); got != "L" {
		t.Errorf("Link = %q, want L", got)
	}
	if got := articleField(text, "Author"); got != "" {
		t.Errorf("Author = %q, want empty", got)
	}
}

// TestRelated checks that the articles of a trip must have a score of at least
// newsMinScore, and that all the articles stay without a trip.
func TestRelated(t *testing.T) {
	docs := func() []internal.Document {
		return []internal.Document{{ID: "near", Score: 0.7}, {ID: "far", Score: 0.5}}
	}
	if got := related(docs(), "Bethesda, MD"); len(got) != 1 || got[0].ID != "near" {
		t.Errorf("related to a trip = %v, want only near", got)
	}
	if got := related(docs(), ""); len(got) != 2 {
		t.Errorf("related without a trip = %v, want both", got)
	}
}

// newsSearch returns a fake SearchFunc that gives saved articles for each
// source, and records the sources that the tool searched.
func newsSearch(saved map[string][]internal.Document, sources *[]string) internal.SearchFunc {
	return func(_ context.Context, collection, source, _ string, n int) ([]internal.Document, error) {
		if collection != newsCollection {
			return nil, fmt.Errorf("collection %q, want news", collection)
		}
		*sources = append(*sources, source)
		d := saved[source]
		return d[:min(n, len(d))], nil
	}
}

// TestGetCurrentNewsSaved checks the three ways to get the news with saved
// articles, so that no feed is read: a search, the feed of an area, and the
// places and the roads of a trip, which give the most related articles of all
// the searches, at most n.
func TestGetCurrentNewsSaved(t *testing.T) {
	savedAt := time.Now().Add(-3 * time.Minute)
	doc := func(id string, score float32) internal.Document {
		return internal.Document{ID: id, Text: article(id, "https://wtop.com/"+id), Score: score, SavedAt: savedAt}
	}
	search := func(q string) string { return fmt.Sprintf(newsSearchURL, url.QueryEscape(q)) }
	saved := map[string][]internal.Document{
		search("Marathon"):             {doc("race", 0)},
		fmt.Sprintf(newsFeedURL, "dc"): {doc("dc news", 0)},
		search("Bethesda, MD"):         {doc("a", 0.9), doc("b", 0.8), doc("c", 0.5)},
		search("I-495"):                {doc("a", 0.95), doc("d", 0.7), doc("e", 0.7), doc("f", 0.7), doc("g", 0.7), doc("h", 0.7)},
	}

	var sources []string
	tool := GetCurrentNews(nil, newsSearch(saved, &sources), time.Minute, 1)
	out, err := run(t, tool, `{"query":"Marathon"}`)
	if n := decodeNews(t, out); err != nil || len(n.Articles) != 1 || n.Saved == nil || *n.Saved != 3 {
		t.Errorf("search = %+v, %v, want the saved article and its age", n, err)
	}
	out, err = run(t, tool, `{"area":"dc"}`)
	if n := decodeNews(t, out); err != nil || !strings.Contains(n.Articles[0], "dc news") {
		t.Errorf("area = %+v, %v, want the dc feed", n, err)
	}

	// n is 1, so the tool gives minNewsResults articles. Article a is in both
	// searches and comes once, and article c is not related.
	out, err = run(t, tool, `{"about":"Bethesda, MD; I-495; "}`)
	n := decodeNews(t, out)
	if err != nil || len(n.Articles) != minNewsResults {
		t.Fatalf("trip = %+v, %v, want %d articles", n, err, minNewsResults)
	}
	if !strings.HasPrefix(n.Articles[0], "Title: a\n") || strings.Contains(out, "Title: c") {
		t.Errorf("trip articles = %q, want a first and no c", n.Articles)
	}
	if n.Saved == nil || *n.Saved != 3 {
		t.Errorf("saved = %v, want 3", n.Saved)
	}
}

// TestGetCurrentNewsErrors checks the errors of the arguments and of the store.
func TestGetCurrentNewsErrors(t *testing.T) {
	var sources []string
	tool := GetCurrentNews(nil, newsSearch(nil, &sources), time.Minute, 5)
	for _, args := range []string{`{`, `{}`, `{"area":"texas"}`} {
		if _, err := run(t, tool, args); err == nil {
			t.Errorf("args %s: no error, want one", args)
		}
	}
	failing := func(context.Context, string, string, string, int) ([]internal.Document, error) {
		return nil, errors.New("down")
	}
	for _, args := range []string{`{"query":"x"}`, `{"about":"Bethesda, MD"}`} {
		if _, err := run(t, GetCurrentNews(nil, failing, time.Minute, 5), args); err == nil || !strings.Contains(err.Error(), "down") {
			t.Errorf("args %s with a failing store: error = %v, want down", args, err)
		}
	}
}

// TestCheckFeedNew checks that a feed that is not saved is read, saved for the
// ttl, and searched for the trip. The feed is a local test server, so that no
// request leaves the computer. A feed that is not RSS is an error.
func TestCheckFeedNew(t *testing.T) {
	rss := `<rss><channel><item><title>Crash on I-495</title><link>https://wtop.com/a</link>` +
		`<pubDate>Mon, 16 Nov 2026 08:00:00 -0500</pubDate><description>Lanes closed</description><category>Traffic</category></item></channel></rss>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/bad" {
			w.Write([]byte("<html"))
			return
		}
		w.Write([]byte(rss))
	}))
	defer srv.Close()

	var saved []internal.Document
	var ttl time.Duration
	save := func(_ context.Context, _ string, docs []internal.Document, d time.Duration) error {
		saved, ttl = docs, d
		return nil
	}
	search := func(_ context.Context, _, _, _ string, _ int) ([]internal.Document, error) {
		docs := slices.Clone(saved)
		for i := range docs {
			docs[i].Score = 0.9
		}
		return docs, nil
	}
	docs, savedAt, err := checkFeed(t.Context(), save, search, time.Minute, 5, srv.URL, "WTOP test", "Bethesda, MD", true)
	if err != nil || len(docs) != 1 || !savedAt.IsZero() {
		t.Fatalf("docs = %v, savedAt = %v, err = %v, want one new article", docs, savedAt, err)
	}
	if ttl != time.Minute || docs[0].ID != "https://wtop.com/a" || !strings.Contains(docs[0].Text, "Summary: Lanes closed") || !strings.Contains(docs[0].Text, "Published: 2026-11-16T08:00:00-05:00") {
		t.Errorf("saved %q for %v, want the article for one minute", docs[0].Text, ttl)
	}

	if _, _, err := checkFeed(t.Context(), save, search, time.Minute, 5, srv.URL+"/bad", "WTOP test", "", true); err == nil {
		t.Error("feed that is not RSS: no error, want one")
	}
	failing := func(context.Context, string, []internal.Document, time.Duration) error { return errors.New("full") }
	if _, _, err := checkFeed(t.Context(), failing, search, time.Minute, 5, srv.URL, "WTOP test", "", true); err == nil {
		t.Error("store fails: no error, want one")
	}
}
