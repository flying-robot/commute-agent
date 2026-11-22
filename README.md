# commute-agent

commute-agent is a local AI agent that plans drives. It uses live traffic,
weather, and news. When the user asks about a trip, it also decides if a
parent can get to the daycare before it closes. If the delay is large, it offers
to send a pickup request to a second person. It sends the request only when the
user accepts.

The agent does an action only when the user sends a message. It does not run
checks on a schedule.

All models run on your computer with Ollama. The agent calls internet services
only for routes, weather, news, and notifications.

## Quick check for evaluators

These steps take about 15 minutes after the models download.

1. Install the prerequisites in the next section.
2. Run the unit tests. They need no models, Docker, or network:

   ```sh
   make test
   ```

   All three packages must show `ok`.
3. Start the agent:

   ```sh
   make
   ```

4. Open <http://localhost:8080> and send the messages in
   [Try the agent](#try-the-agent).
5. Optional: test the pickup flow in [Test the pickup flow](#test-the-pickup-flow).
6. Optional: run the eval in [Run the eval](#run-the-eval). It takes about one
   hour.

## Prerequisites

- Go 1.27 or later
- [Ollama](https://ollama.com), running
- Docker, running (for the Chroma vector database)
- `make`, `curl`, and `jq`
- About 12 GB of free memory for the two chat models

Pull the models one time:

```sh
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

The `Makefile` requires a TomTom evaluation key which has been provided in the submission.

## What to check

| Feature | How to see it | Code |
| --- | --- | --- |
| Multi-agent flow | Each answer on the chat page shows the steps of three agents: the trip planner (`qwen3.5:9b`) gets the route, the conditions agent (`qwen3.5:4b`) gets the weather and the news, and the reporter (`qwen3.5:4b`) writes the answer. | `newTurn` and `runAgent` in `internal/functions.go` |
| Tool use | The chat page shows each tool call and its result. There are 9 tools. | `internal/tools/` |
| Memory | Save a fact, then ask about it in a new conversation. Facts go to `memory.md`. | `save_memory_tool.go`, `search_memory_tool.go`, `forget_memory_tool.go` |
| Retrieval | News and weather results go to a Chroma vector store. Ask `What is saved?` to see them. | `chroma_provider.go`, `search_saved_data_tool.go` |
| Decision logic | The answer says to leave on time, leave earlier, or take a different route, from the size of the delay. | `decide` in `internal/functions.go` |
| Human approval | The agent asks before it sends a pickup request, and sends it one time only. | [Test the pickup flow](#test-the-pickup-flow) |
| Scope limit | A question that is not about travel gets a one-sentence refusal and no tool call. | `plannerPrompt` in `internal/functions.go` |
| Prompt-injection guard | After a tool gives text from the internet, the tools that change the memory fail for the rest of the conversation. | `untrustedTools` and `writeTools` in `internal/functions.go` |
| Failure handling | If a notification fails, the chat says that the request was NOT sent. | `NOTIFY_FAIL` in [Test the pickup flow](#test-the-pickup-flow) |
| Tests | Unit tests with fake models and services, and an eval with the real models. | `*_test.go`, `internal/eval_test.go` |

## Try the agent

Start the agent with `make`. This starts Chroma in a Docker container named
`commute-agent-chroma`, and then the server. Open <http://localhost:8080>.

The first answer is slower, because Ollama loads the models.

Send these messages:

| Message | Expected result |
| --- | --- |
| `How long is the drive from the White House to the Pentagon right now?` | A travel time with live traffic, weather, and news |
| `When must I leave Union Station to arrive at Dulles Airport by 9:00 tomorrow?` | A departure time |
| `Remember that my home is 1600 Pennsylvania Ave NW, Washington, DC.` | The fact is saved in `memory.md` |
| `What is saved?` | A list of the saved news and weather, with the age of each one |
| `Write me a poem about the sea.` | A refusal, because the question is not about travel |

You can also send a message from a second terminal:

```sh
make chat MESSAGE="How is the drive from Baltimore to Annapolis now?"
```

Stop the server with Ctrl+C. Stop Chroma with `make chroma-stop`.

## Test the pickup flow

The pickup flow needs a pickup plan in `memory.md`, in the project folder.

1. Create `memory.md` with this content:

   ```markdown
   ## pickup

   name: Alice
   start: 11725 Berry Rd, Waldorf, MD 20603
   destination: 3401 Richmond Hwy, Alexandria, VA 22305
   leave_by: 16:45
   closes: 18:00
   days: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
   contact: Bob
   ```

2. Start the server with forced delegation, so that you do not need a real
   traffic delay:

   ```sh
   make FORCE_DELEGATION=true
   ```

3. Open <https://ntfy.sh/commute-agent-demo-notifs-0118> in a browser tab to
   see the notifications. The title of each one says whom it is for.
4. In the chat, ask about the drive, for example `How is the drive from Waldorf
   to Alexandria now?` The answer ends with "Do you want me to ask Bob to do
   the pickup today?"
5. Reply `yes`. A notification "For Bob: Pickup request" arrives, and the chat
   shows the message that was sent. A second `yes` does not send it again.

To test a failed notification, start the server with
`make FORCE_DELEGATION=true NOTIFY_FAIL=error` (or `timeout` or `rate_limit`).
The chat then says that the request was NOT sent, and tells the user to call
the contact.

The server reads `memory.md` when it first needs it. Restart the server after
you change the file. A restart also clears the pending offers and the "sent
today" record.

## Run the eval

```sh
make eval EVAL_RUNS=3
```

The eval runs fixed conversations with the real models and the real APIs. It
checks the tool calls and the answers. The cases include routes, memory, news,
weather, prompt injection, each agent alone, fallbacks when a service fails,
questions that are not about travel, and the delay decisions. The case list is
in `internal/eval_test.go`.

The eval needs Ollama and the network, but not Chroma or Docker. It took 53
minutes with `EVAL_RUNS=3` on the test computer. It sends no real notification.

## Configuration

Set a variable on the `make` command line, for example `make MAJOR_DELAY=20`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_NAME` | `qwen3.5:9b` | Model of the trip planner |
| `SMALL_MODEL_NAME` | `qwen3.5:4b` | Model of the conditions agent and the reporter |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `SERVER_ADDR` | `localhost:8080` | Address of the chat page and the API |
| `CHROMA_ADDR` | `127.0.0.1:8000` | Address of Chroma |
| `MEMORY_FILE` | `memory.md` | Saved facts and the pickup plan |
| `FEEDBACK_FILE` | `feedback.md` | Trip feedback |
| `DATA_TTL` | `10m` | How long news and weather results are reused |
| `NEWS_RESULTS` | `5` | Number of related articles for each trip |
| `MINOR_DELAY` | `15` | Largest delay, in minutes, for "leave earlier" |
| `MAJOR_DELAY` | `30` | Largest delay, in minutes, for "take a different route" |
| `FORCE_DELEGATION` | `false` | Offer delegation for each route, for testing |
| `NTFY_TOPIC` | `commute-agent-demo-notifs-0118` | ntfy.sh topic for notifications |
| `NOTIFY_FAIL` | empty | Inject a notification failure: `error`, `timeout`, or `rate_limit` |
| `TOMTOM_API_KEY` | evaluation key | TomTom routing key |

Anyone who knows the ntfy.sh topic can read it, and the pickup requests have
the destination address. For real use, use a topic that is hard to guess.

## Code layout

The domain logic gets all side effects, such as model calls and HTTP requests,
through function types in one `Actions` structure. Thus the unit tests can
replace each external system with a fake.

| Path | Contents |
| --- | --- |
| `cmd/commute-agent/main.go` | Reads the configuration, builds the providers, and starts the agent |
| `internal/types.go` | Data types, function types, and the `Actions` structure |
| `internal/functions.go` | Prompts, agent loop, coordinator, decider, pickup offers |
| `internal/providers/` | One file for each external system: Ollama, Chroma, file store, ntfy.sh, HTTP server, clock, log |
| `internal/tools/` | One file for each tool that a model can call |
| `internal/eval_test.go` | The eval suite |
