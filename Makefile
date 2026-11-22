MODEL_NAME       ?= qwen3.5:9b
SMALL_MODEL_NAME ?= qwen3.5:4b
EMBED_MODEL      ?= nomic-embed-text
SERVER_ADDR      ?= localhost:8080
MEMORY_FILE      ?= memory.md
CHROMA_ADDR      ?= 127.0.0.1:8000
MESSAGE          ?= What can you help me with?
FEEDBACK_FILE    ?= feedback.md
DATA_TTL         ?= 10m
NEWS_RESULTS     ?= 5
MINOR_DELAY      ?= 15
MAJOR_DELAY      ?= 30
FORCE_DELEGATION ?= false
NTFY_TOPIC       ?= commute-agent-demo-notifs-0118
NOTIFY_FAIL      ?=
TOMTOM_API_KEY   ?=

.EXPORT_ALL_VARIABLES:

.PHONY: all
all: chroma
	@go run cmd/commute-agent/main.go

# Start the Chroma vector database in Docker at CHROMA_ADDR, and wait until it
# answers. The first run makes the container, and later runs start it again with
# its data. Stop it with "make chroma-stop".
.PHONY: chroma
chroma:
	@docker start commute-agent-chroma >/dev/null 2>&1 \
		|| docker run -d --name commute-agent-chroma -p $(CHROMA_ADDR):8000 chromadb/chroma:1.5.9 >/dev/null
	@until curl -sf http://$(CHROMA_ADDR)/api/v2/heartbeat >/dev/null; do sleep 1; done

# Stop the Chroma container. The container and its data stay, so that "make" or
# "make chroma" can start it again.
.PHONY: chroma-stop
chroma-stop:
	@docker stop commute-agent-chroma >/dev/null

# Send MESSAGE to the running server and format the reply with jq. Start the
# server first with "make" in a different terminal. For example:
#   make chat MESSAGE="What is the best time to leave for work?"
# The conversation cookie is kept in .chat-cookies, so that the messages are in
# one conversation. Delete the file to start a new conversation.
.PHONY: chat
chat:
	@jq -n --arg content "$(MESSAGE)" '{content: $$content}' \
		| curl -sS -b .chat-cookies -c .chat-cookies -X POST http://$(SERVER_ADDR)/messages -H 'Content-Type: application/json' -d @- \
		| jq .

# Run the eval: fixed conversations with the real model and the real APIs, with
# checks of the tool calls of the agent. It takes several minutes. Set EVAL_RUNS
# to run each conversation more than once, for example:
#   make eval EVAL_RUNS=3
.PHONY: eval
eval:
	@go test -tags eval -run TestEval -count=1 -timeout 60m -json ./internal/ \
		| jq -r --unbuffered ' \
			if .Action == "run" then \
				"=== RUN   \(.Test)" \
			elif .Action == "pass" or .Action == "fail" then \
				"--- \(.Action | ascii_upcase): \(.Test // .Package) (\(.Elapsed)s)" \
			elif .Action == "build-output" or (.Action == "output" and (.Output | test("^\\s+eval_test"))) then \
				.Output | rtrimstr("\n") \
			else empty end'

.PHONY: test
test:
	@go test -cover -v ./...

.PHONY: claude
claude:
	@claude --resume commute-agent
