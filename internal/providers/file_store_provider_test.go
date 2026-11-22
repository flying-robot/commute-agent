package providers

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/flying-robot/commute-agent/internal"
)

// TestFileStore checks that a save writes the Markdown file, that a new store
// reads the file again after a restart, and that a search and a delete use it.
func TestFileStore(t *testing.T) {
	ctx := t.Context()
	path := filepath.Join(t.TempDir(), "memory.md")
	save, search, _ := NewFileStoreProvider(path, letterEmbed)

	// A missing file means that there are no documents yet.
	if docs, err := search(ctx, "memory", "", "", 10); err != nil || len(docs) != 0 {
		t.Fatalf("search of a missing file = %v, %v, want none", ids(docs), err)
	}
	if err := save(ctx, "memory", []internal.Document{{ID: "work", Text: "work: Waldorf"}, {ID: "home", Text: "home: Alexandria"}}, 0); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if want := "## home\n\nhome: Alexandria\n\n## work\n\nwork: Waldorf\n"; string(b) != want {
		t.Errorf("file =\n%s\nwant\n%s", b, want)
	}

	// A new store, as after a restart, reads the facts from the file.
	_, search2, del2 := NewFileStoreProvider(path, letterEmbed)
	docs, err := search2(ctx, "memory", "", "work", 1)
	if err != nil || len(docs) != 1 || docs[0].ID != "work" {
		t.Fatalf("search after a restart = %v, %v, want work", ids(docs), err)
	}
	if ok, err := del2(ctx, "memory", "work"); !ok || err != nil {
		t.Errorf("delete = %v, %v, want true", ok, err)
	}
	if ok, err := del2(ctx, "memory", "work"); ok || err != nil {
		t.Errorf("second delete = %v, %v, want false", ok, err)
	}
	b, _ = os.ReadFile(path)
	if strings.Contains(string(b), "work") {
		t.Errorf("file after the delete =\n%s\nwant no work", b)
	}
}

// TestFileStoreErrors checks the errors of the file store: a save with a ttl,
// an empty path, a failed embedding, and a file that is not readable.
func TestFileStoreErrors(t *testing.T) {
	ctx := t.Context()
	doc := []internal.Document{{ID: "a", Text: "a"}}

	save, _, _ := NewFileStoreProvider(filepath.Join(t.TempDir(), "m.md"), letterEmbed)
	if err := save(ctx, "memory", doc, 1); err == nil {
		t.Error("save with a ttl: no error, want one")
	}

	save, search, del := NewFileStoreProvider("", letterEmbed)
	if err := save(ctx, "memory", doc, 0); err == nil {
		t.Error("save with no path: no error, want one")
	}
	if _, err := search(ctx, "memory", "", "", 1); err == nil {
		t.Error("search with no path: no error, want one")
	}
	if _, err := del(ctx, "memory", "a"); err == nil {
		t.Error("delete with no path: no error, want one")
	}

	failing := func(context.Context, []string, bool) ([][]float32, error) { return nil, errors.New("broken") }
	path := filepath.Join(t.TempDir(), "m.md")
	os.WriteFile(path, []byte("## a\n\na\n"), 0o600)
	_, search, _ = NewFileStoreProvider(path, failing)
	if _, err := search(ctx, "memory", "", "", 1); err == nil {
		t.Error("search with a failed embedding: no error, want one")
	}

	// A directory is not a file that the store can read.
	_, search, _ = NewFileStoreProvider(t.TempDir(), letterEmbed)
	if _, err := search(ctx, "memory", "", "", 1); err == nil {
		t.Error("search of a directory: no error, want one")
	}
}

// TestParseMarkdown checks that each "## " heading starts a document, that the
// text under it is trimmed, and that lines before the first heading are not
// used.
func TestParseMarkdown(t *testing.T) {
	got := parseMarkdown("intro\n## a\n\nline 1\nline 2\n\n## b\nb\n")
	if len(got) != 2 || got["a"] != "line 1\nline 2" || got["b"] != "b" {
		t.Errorf("parseMarkdown = %q, want a and b", got)
	}
	if got := parseMarkdown(""); len(got) != 0 {
		t.Errorf("parseMarkdown of nothing = %q, want none", got)
	}
	texts := map[string]string{"b": "2", "a": "1"}
	if got := string(formatMarkdown(texts)); got != "## a\n\n1\n\n## b\n\n2\n" {
		t.Errorf("formatMarkdown = %q", got)
	}
	if back := parseMarkdown(string(formatMarkdown(texts))); !slices.Equal([]string{back["a"], back["b"]}, []string{"1", "2"}) {
		t.Errorf("parse of format = %q, want the same texts", back)
	}
}

// TestReplaceFile checks that replaceFile writes the data, makes a file that
// only the user can read, and leaves no temporary file. A directory that does
// not exist is an error.
func TestReplaceFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.md")
	if err := replaceFile(path, []byte("one")); err != nil {
		t.Fatal(err)
	}
	if err := replaceFile(path, []byte("two")); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	info, _ := os.Stat(path)
	entries, _ := os.ReadDir(dir)
	if string(b) != "two" || info.Mode().Perm() != 0o600 || len(entries) != 1 {
		t.Errorf("data = %q, mode = %v, files = %d, want two, 0600, 1", b, info.Mode().Perm(), len(entries))
	}
	if err := replaceFile(filepath.Join(dir, "missing", "m.md"), nil); err == nil {
		t.Error("replaceFile in a missing directory: no error, want one")
	}
}
