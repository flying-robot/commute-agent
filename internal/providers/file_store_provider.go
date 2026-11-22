package providers

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/flying-robot/commute-agent/internal"
)

// fileCollection is the collection of the in-memory store that has the documents
// of the file.
const fileCollection = "file"

// NewFileStoreProvider returns an internal.SaveFunc, an internal.SearchFunc, and
// an internal.DeleteFunc that keep documents in a Markdown file at path, so that
// they stay after a restart. Each document is a "## " heading with its ID, and
// the text under it, so that a person can read and edit the file. It has no vectors, so the provider embeds the texts when
// it first reads the file, and a different embedding model needs no change to
// the file.
//
// A file has one collection, so the functions do not use their collection
// argument. The documents stay until they are deleted, so a save with a ttl
// fails, and the source and the save time of a document are not kept.
func NewFileStoreProvider(path string, embed internal.EmbedFunc) (internal.SaveFunc, internal.SearchFunc, internal.DeleteFunc) {
	// The in-memory store does the embedding and the search. The mutex makes
	// sure that two calls do not change the texts or the file at the same time.
	// The texts are nil until the file is read.
	innerSave, innerSearch, innerDelete := NewVectorStoreProvider(embed)
	var mu sync.Mutex
	var texts map[string]string

	// load reads the file and embeds its texts, the first time that a function
	// needs them. A missing file means that there are no documents yet. If the
	// load fails, the next call tries again. The caller must hold the mutex.
	load := func(ctx context.Context) error {
		if texts != nil {
			return nil
		}
		if path == "" {
			return errors.New("the path of the file store is not set")
		}
		b, err := os.ReadFile(path)
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("failed to read %s: %w", path, err)
		}
		m := parseMarkdown(string(b))
		var docs []internal.Document
		for id, text := range m {
			docs = append(docs, internal.Document{ID: id, Text: text})
		}
		if err := innerSave(ctx, fileCollection, docs, 0); err != nil {
			return err
		}
		texts = m
		return nil
	}

	// write replaces the file with the texts. The caller must hold the mutex.
	write := func() error {
		if err := replaceFile(path, formatMarkdown(texts)); err != nil {
			return fmt.Errorf("failed to write %s: %w", path, err)
		}
		return nil
	}

	save := func(ctx context.Context, _ string, docs []internal.Document, ttl time.Duration) error {
		// First, embed the documents in the in-memory store, because that is the
		// step that fails most often. Then write the file.
		if ttl > 0 {
			return errors.New("a file store keeps documents until they are deleted, so it has no ttl")
		}
		mu.Lock()
		defer mu.Unlock()
		if err := load(ctx); err != nil {
			return err
		}
		if err := innerSave(ctx, fileCollection, docs, 0); err != nil {
			return err
		}
		for _, d := range docs {
			texts[d.ID] = d.Text
		}
		return write()
	}

	search := func(ctx context.Context, _, source, query string, n int) ([]internal.Document, error) {
		mu.Lock()
		defer mu.Unlock()
		if err := load(ctx); err != nil {
			return nil, err
		}
		return innerSearch(ctx, fileCollection, source, query, n)
	}

	del := func(ctx context.Context, _, id string) (bool, error) {
		mu.Lock()
		defer mu.Unlock()
		if err := load(ctx); err != nil {
			return false, err
		}
		ok, err := innerDelete(ctx, fileCollection, id)
		if err != nil || !ok {
			return ok, err
		}
		delete(texts, id)
		return true, write()
	}

	return save, search, del
}

// parseMarkdown returns the IDs and the texts of a Markdown file. Each line that
// starts with "## " is the ID of a document, and the lines under it, up to the
// next such line, are its text. Lines before the first ID are not used.
func parseMarkdown(s string) map[string]string {
	m := map[string]string{}
	var id, text string
	add := func() {
		if id != "" {
			m[id] = strings.TrimSpace(text)
		}
	}
	for line := range strings.Lines(s) {
		if h, ok := strings.CutPrefix(line, "## "); ok {
			add()
			id, text = strings.TrimSpace(h), ""
			continue
		}
		text += line
	}
	add()
	return m
}

// formatMarkdown returns the Markdown file of the texts, in the order of their
// IDs, so that a save changes only the lines of its document.
func formatMarkdown(texts map[string]string) []byte {
	var b strings.Builder
	for i, id := range slices.Sorted(maps.Keys(texts)) {
		if i > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "## %s\n\n%s\n", id, texts[id])
	}
	return []byte(b.String())
}

// replaceFile replaces the file at path with data. It writes the data to a
// temporary file in the same directory first, and then renames it to path. A
// rename in one directory is atomic, so if the program stops during the write,
// the old file stays complete and only the temporary file is left. The temporary
// file is created so that only the user can read it, because the facts can have
// home and work addresses.
func replaceFile(path string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+"-*")
	if err != nil {
		return err
	}
	// Remove the temporary file if a step fails. After the rename, the name no
	// longer exists, so this does nothing.
	defer os.Remove(tmp.Name())

	// Write the data, and make sure that it is on the disk before the rename, so
	// that a crash after the rename does not leave an empty file.
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
