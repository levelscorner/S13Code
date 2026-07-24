# S13Code

`S13Code` is the standalone Session 13 agent runtime. It implements a live task graph, scoped and provenance-bearing memory, Rohan's semantic chunking V2, and Agent2Agent interoperability. It asks `glc_v3` for model completions over HTTP and never owns provider credentials.

## What runs where

| Service | Default address | Responsibility |
|---|---|---|
| `glc_v3` | `http://127.0.0.1:8111` | Models, keys, routing and channels |
| `S13Code` HTTP | `http://127.0.0.1:8113` | Graph, memory, documents and JSON-RPC A2A |
| `S13Code` gRPC | `127.0.0.1:8114` | Official A2A gRPC service |
| Ollama | `http://127.0.0.1:11434` | Phi-4 segmentation and Nomic embeddings |

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A running `glc_v3`
- A running Ollama with `phi4` and `nomic-embed-text`

```bash
ollama pull phi4
ollama pull nomic-embed-text
ollama serve
```

## Install and run

Unzip `glc_v3`, `S13Code`, and `S13Proof` beside one another. Start `glc_v3` first. Then, from this directory:

```bash
uv sync

export GLC_BASE_URL=http://127.0.0.1:8111
export S13_GATEWAY_PROVIDER=gemini
export S13_SANDBOX_ROOT="$PWD/sandbox"
export S13_CHUNK_MODEL=phi4:latest
export S13_LIVE_SEMANTIC_CHUNKING=1

uv run s13code serve
```

State is written under `~/.s13code` by default. Set `S13_DATA_DIR` to use another directory.

Check both services:

```bash
curl http://127.0.0.1:8111/healthz
curl http://127.0.0.1:8113/healthz
curl http://127.0.0.1:8113/readyz
curl http://127.0.0.1:8113/.well-known/agent-card.json
```

## Run a prompt

```bash
curl -s http://127.0.0.1:8113/v1/agent/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "course",
    "project_id": "s13",
    "user_id": "student-01",
    "agent_id": "assistant",
    "prompt": "Say hello."
  }'
```

The response contains the final answer, graph nodes and edges, ordered graph events, and provider/agent assignments. Inspect a persisted run with:

```bash
curl http://127.0.0.1:8113/v1/agent/runs/<run-id>
```

## Index the sample corpus

The five files under `sandbox/papers/` are fixed `.txt` fixtures for semantic chunking and retrieval proofs.

```bash
curl -s http://127.0.0.1:8113/v1/agent/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "course",
    "project_id": "papers",
    "user_id": "student-01",
    "prompt": "Index every .txt file under papers/. Confirm how many chunks were indexed in total."
  }'
```

Document ingestion is versioned and atomic: source preparation, semantic boundaries, exact spans, Nomic embeddings, and visibility succeed together or roll back together.

## Architecture

- `s13code/core/live_graph/`: durable graph state, patches, event replay and bounded parallel execution
- `s13code/core/memory/`: scope checks, provenance, contradiction history, semantic chunking and FAISS retrieval
- `s13code/core/a2a_adapter/`: Agent Cards, JSON-RPC, SSE/push, official gRPC and trust checks
- `s13code/gateway.py`: the only `S13Code → glc_v3` seam
- `s13code/runtime.py`: joins graph, memory, tools and model calls into an inspectable run
- `tests/`: executable invariants and regression cases

## Test before opening a pull request

```bash
uv run ruff check .
uv run pytest -q

cd ../S13Proof
uv sync
uv run pytest -q
```

## Student contribution

Fork the official [`theschoolofai/S13Code`](https://github.com/theschoolofai/S13Code) repository linked from Axiom, create a branch, implement one meaningful extension, and open one pull request against that repository. Do not open the Session 13 pull request against [`theschoolofai/glc_v3`](https://github.com/theschoolofai/glc_v3).

Add one subsection to this README in the same pull request. It must contain:

1. the user-visible capability,
2. the exact prompt or API request,
3. the graph and ordered event trace,
4. the actual final result,
5. evidence and provider/agent assignments,
6. the adversarial failure and its fix,
7. commands that reproduce the result from a fresh checkout.

Do not commit `.env`, credentials, personal memory, generated databases, unrestricted local paths, benchmark output containing private data, or provider responses containing secrets. Use synthetic identities in every proof.

### Budget-aware branch: a per-run cap on task launches

**1. User-visible capability**

A run can now be given a hard cap on how many tasks it may launch. When the cap is
reached the agent stops buying new work, drops the backlog it has not started, and
answers from the evidence it already holds — instead of either overspending or
returning nothing.

Before this change the live graph had no per-run limit of any kind. The planner
expands the frontier from each outcome, so one request could launch an unbounded
number of tasks, each an LLM or embedding call. Session 12 invariant 8 requires
hard per-run limits on time, tokens, tool calls and cost; the graph enforced none.
`RunBudget` supplies the tool-call limit.

The cap is opt-in. Unset, the runtime behaves exactly as before.

**2. Exact API request**

Serve with the cap set (full environment in item 7), then:

```bash
curl -s http://127.0.0.1:8113/v1/agent/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "course",
    "project_id": "papers",
    "user_id": "student-01",
    "agent_id": "assistant",
    "prompt": "Index every .txt file under papers/. Confirm how many chunks were indexed in total."
  }'
```

`sandbox/papers/` holds five `.txt` fixtures, so the honest cost of this prompt is
one directory listing plus five index tasks. The run below was capped at three
launches.

**3. Graph and ordered event trace**

Run `run-c69b03ee08bc`, `S13_MAX_TASK_LAUNCHES=3`.

Nodes:

```
list_directory  succeeded
index_1         succeeded
index_2         cancelled   <- cancelled while still running
index_3         cancelled   <- never launched
index_4         cancelled   <- never launched
index_5         cancelled   <- never launched
answer          succeeded
```

Edges into the answer: `list_directory -> answer`, `index_1 -> answer`. No cancelled
node is connected to the answer.

Ordered events:

```
 1  run_started
 2  graph_patched
 3  task_started      list_directory
 4  task_succeeded    list_directory     found 5 files
 5  graph_patched                        index_1..index_5 added
 6  task_started      index_1
 7  task_started      index_2            third launch: the cap is now reached
 8  task_succeeded    index_1            4 semantic chunks
 9  task_cancelled    index_2            was_running=true
10  task_cancelled    index_3            was_running=false
11  task_cancelled    index_4            was_running=false
12  task_cancelled    index_5            was_running=false
13  budget_exhausted                     launches=3 max_task_launches=3
                                         cancelled=[index_2, index_3, index_4, index_5]
14  graph_patched                        answer added
15  task_started      answer
16  task_succeeded    answer
17  graph_patched                        run finished
```

Events 6 and 7 are the only launches the cap allowed after `list_directory`.
Events 9 through 12 are one transaction with event 8: the outcome that reached the
cap and the cancellations it forced commit together. `index_3` through `index_5`
have no `task_started` at all — the cap is enforced before work begins, not after
it has been paid for.

**4. Actual final result**

```
Based on the authorized memory evidence, 4 semantic chunks were indexed from the
document located at `file://<repo>/sandbox/papers/attention.txt`.

To confirm how many chunks were indexed in total from all `.txt` files under the
`papers/` directory, additional information about other files and their respective
index reports would be required. Since only one file's indexing report is provided,
I cannot determine the total number of chunks indexed across all files without
further evidence.
```

(The absolute checkout path in the model's answer is redacted to `<repo>`.)

The run ends `completed` with a real answer. The model reports only what the
budget actually bought — one file, four chunks — and declines to state a corpus
total it cannot support. Partial evidence produced a truthful partial answer
rather than a confident wrong one.

**5. Evidence and provider/agent assignments**

`evidence_count` on the answer node is `1`: only the succeeded `index_1` contributed.
The four cancelled tasks contributed nothing to retrieval or to the answer.

| node | agent / skill | state | provider | model |
|---|---|---|---|---|
| `list_directory` | `list_directory` | succeeded | — (local tool) | — |
| `index_1` | `index_file` | succeeded | — (local chunking + Nomic embeddings) | — |
| `index_2`–`index_5` | `index_file` | cancelled | — (never billed a provider) | — |
| `answer` | `answer_with_evidence` | succeeded | `ollama` via `glc_v3` | `phi4:latest` |

Planner mode: `deterministic`. Every model call crossed `glc_v3`; `S13Code` holds no
provider credentials.

Same prompt, same fixtures, measured both ways:

| | task launches | cancelled | evidence | wall clock | `budget_exhausted` |
|---|---|---|---|---|---|
| default (unlimited) | 7 | 0 | 5 | 119s | none |
| `S13_MAX_TASK_LAUNCHES=3` | 3 | 4 | 1 | 40s | 1 |

Both runs end `completed` with an answer. The cap traded corpus coverage for a
bounded, predictable spend — which is the decision the operator asked for.

**6. The adversarial failure and its fix**

Cancelling the backlog is easy when everything is pending. The dangerous case is the
one this design creates: the cap is reached by an outcome while a **sibling task is
still running**. That task is cancelled mid-flight, and its result arrives afterwards.
If that late result were recorded, work the budget explicitly refused would re-enter
the graph and reach the answer — the cap would be cosmetic.

The trace above hits this case for real: `index_2` was cancelled with
`was_running=true`.

`s13code/core/live_graph/tests/test_budget_aware_branch.py::test_adversarial_late_result_from_budget_cancelled_task_never_lands`
launches two tasks together, exhausts the cap with the first, and asserts the second
never lands: no `task_succeeded` event, `result is None`, its payload absent from the
answer, and the run still finalises.

Removing the two enforcement points (the `ready()` clamp and the `_enforce_budget`
call in `record_outcome`) reproduces the failure behaviourally:

```
assert snapshot.nodes["w2"]["state"] == "cancelled"
E    AssertionError: assert 'succeeded' == 'cancelled'

assert launched == ["w1", "w2"]
E    AssertionError: assert ['w1', 'w2', 'w3', 'w4'] == ['w1', 'w2']
```

The fix has two halves, and both are needed:

- `ready()` clamps the batch it hands out to the launches still permitted, so the cap
  is enforced *before* work starts. Clamping alone is not enough: it would leave the
  run with a permanently empty frontier and no answer.
- `record_outcome()` cancels the remaining backlog — pending, waiting *and* running —
  and journals `budget_exhausted`, inside the same transaction as the outcome that
  reached the cap. Because that happens before the executor consults the planner, the
  planner sees an all-terminal graph and finalises. The already-present executor rule
  that discards a result whose node is `CANCELLED` then guarantees the late result
  from `index_2` is dropped.

Once exhaustion is journalled the clamp stops gating, so the planner's finalisation
step is free to run. That is deliberate: a run that spends its whole budget and
returns nothing is strictly worse than one that answers from partial evidence.

The launch count is derived by counting `task_started` events, so the journal stays
the single source of truth and the database schema is unchanged. Crash replay is
unaffected — a resumed run recounts its own launches from the journal.

**Honest limitation.** The budget counts task *launches*, not tokens, wall-clock or
currency. Two runs with the same launch count can cost very differently, so this is a
bounded proxy for spend and not a spend meter; tokens and time still need their own
limits. It also does not refund a task cancelled in flight — the provider call may
already be under way, so the budget prevents *further* launches and prevents the
partial result from being used, but tokens already emitted are sunk. Finally, the
finalisation step runs after the cap is reached, so the true worst case is N work
launches plus one answer.

**7. Reproduce from a fresh checkout**

```bash
# Unzip glc_v3, S13Code and S13Proof beside one another.
# Start glc_v3 first on 127.0.0.1:8111 (see its README).
# The proof below runs fully locally: Ollama with phi4:latest and
# nomic-embed-text. No provider API keys are required.

git clone https://github.com/levelscorner/S13Code.git
cd S13Code
git checkout s13-smart-failure
uv sync

uv run ruff check .        # All checks passed!
uv run pytest -q           # 49 passed  (44 upstream + 5 new)

# just this feature's tests, including the adversarial case
uv run pytest s13code/core/live_graph/tests/test_budget_aware_branch.py -q

export GLC_BASE_URL=http://127.0.0.1:8111
export S13_GATEWAY_PROVIDER=ollama
export S13_SANDBOX_ROOT="$PWD/sandbox"
export S13_CHUNK_MODEL=phi4:latest
export S13_LIVE_SEMANTIC_CHUNKING=1
export S13_MAX_TASK_LAUNCHES=3      # the new knob; unset means unlimited
uv run s13code serve
```

In a second shell, send the request from item 2. Omit `S13_MAX_TASK_LAUNCHES` and
send it again to see the unbudgeted run in the comparison table.

Verified against `S13Proof` on the same build: `uv run python run_benchmark.py
--base-url http://127.0.0.1:8113` completed all 14 benchmark cases with answers.

## License

MIT. See `LICENSE`.
