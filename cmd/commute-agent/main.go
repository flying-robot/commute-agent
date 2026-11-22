package main

import (
	"context"
	"os"
	"os/signal"
	"strconv"
	"time"

	"github.com/flying-robot/commute-agent/internal"
	"github.com/flying-robot/commute-agent/internal/providers"
	"github.com/flying-robot/commute-agent/internal/tools"
)

func main() {
	// Set up a context that will be canceled when an interrupt signal is received.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Create a new structured JSON logger using the slog provider. This will be
	// reused throughout the application.
	log := providers.NewSlogProvider(os.Stdout)

	// Create a new Ollama chat provider. This configures the actual model and API
	// that we'll use when generating responses. A second, smaller model does the
	// simpler tasks faster.
	chat, err := providers.NewOllamaProvider(os.Getenv("MODEL_NAME"))
	if err != nil {
		log(ctx, internal.LevelError, "failed to create Ollama provider", "error", err)
		os.Exit(1)
	}
	smallChat, err := providers.NewOllamaProvider(os.Getenv("SMALL_MODEL_NAME"))
	if err != nil {
		log(ctx, internal.LevelError, "failed to create the small Ollama provider", "error", err)
		os.Exit(1)
	}

	// Create a vector store in Chroma, with embeddings from an Ollama embedding
	// model, so that the tools can save the news and the weather and the model
	// can search them by meaning.
	embed, err := providers.NewOllamaEmbedProvider(os.Getenv("EMBED_MODEL"))
	if err != nil {
		log(ctx, internal.LevelError, "failed to create Ollama embedding provider", "error", err)
		os.Exit(1)
	}
	save, search := providers.NewChromaProvider(os.Getenv("CHROMA_ADDR"), embed)

	// Create a store on disk for the facts that the user asks the agent to
	// remember, so that they stay after a restart.
	remember, recall, forget := providers.NewFileStoreProvider(os.Getenv("MEMORY_FILE"), embed)

	// Create a store on disk for the feedback after each trip, which refines the
	// later estimates.
	saveTrip, trips, _ := providers.NewFileStoreProvider(os.Getenv("FEEDBACK_FILE"), embed)

	// Create the provider for the notifications, which publishes them to a topic
	// on ntfy.sh. NOTIFY_FAIL can inject a failure into it instead.
	notify, err := providers.NewNtfyProvider(os.Getenv("NTFY_TOPIC"), os.Getenv("NOTIFY_FAIL"))
	if err != nil {
		log(ctx, internal.LevelError, "failed to create the ntfy.sh provider", "error", err)
		os.Exit(1)
	}

	// Read how long the news and the weather stay saved, and how many news
	// articles a trip gets.
	ttl, err := time.ParseDuration(os.Getenv("DATA_TTL"))
	if err != nil {
		log(ctx, internal.LevelError, "DATA_TTL is not a duration", "error", err)
		os.Exit(1)
	}
	newsResults, err := strconv.Atoi(os.Getenv("NEWS_RESULTS"))
	if err != nil {
		log(ctx, internal.LevelError, "NEWS_RESULTS is not a number", "error", err)
		os.Exit(1)
	}

	// Read the thresholds of the decider in minutes.
	minorDelay, err := strconv.Atoi(os.Getenv("MINOR_DELAY"))
	if err != nil {
		log(ctx, internal.LevelError, "MINOR_DELAY is not a number", "error", err)
		os.Exit(1)
	}
	majorDelay, err := strconv.Atoi(os.Getenv("MAJOR_DELAY"))
	if err != nil {
		log(ctx, internal.LevelError, "MAJOR_DELAY is not a number", "error", err)
		os.Exit(1)
	}

	// Read whether each route needs a delegated pickup, to test the delegation.
	forceDelegation, err := strconv.ParseBool(os.Getenv("FORCE_DELEGATION"))
	if err != nil {
		log(ctx, internal.LevelError, "FORCE_DELEGATION is not true or false", "error", err)
		os.Exit(1)
	}

	// Compose the agent by injecting impure actions into it, and run it. The
	// trip planner and the conditions agent each get their own tools. The agent
	// is a RESTful API, with the map of each route that the get_route tool finds,
	// and it runs the scheduled pickup checks, until an interrupt signal cancels
	// the context.
	planner, conditions := tools.All(os.Getenv("TOMTOM_API_KEY"), save, search, ttl, newsResults, remember, recall, forget)
	err = internal.Run(ctx, internal.Actions{
		Chat:            chat,
		SmallChat:       smallChat,
		Log:             log,
		Now:             providers.NewClockProvider(),
		Sleep:           providers.NewSleepProvider(),
		Planner:         planner,
		Conditions:      conditions,
		Notify:          notify,
		SaveTrip:        saveTrip,
		Trips:           trips,
		Serve:           providers.NewHTTPProvider(os.Getenv("SERVER_ADDR"), tools.RouteMaps()),
		MinorDelay:      minorDelay,
		MajorDelay:      majorDelay,
		ForceDelegation: forceDelegation,
	})
	if err != nil {
		log(ctx, internal.LevelError, "server failed", "error", err)
		os.Exit(1)
	}
}
