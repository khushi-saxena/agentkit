# agentkit

A six-agent pipeline that reads a developer's GitHub and resume, works out what
they are actually good at, decides what they should learn next, and finds real
open-source issues that both fit and stretch them — plus the SDK it was
refactored into, so the same blocks power other products.

Built on LangChain and LangGraph, with a FastAPI + SSE frontend that streams
each agent's result as it finishes.

```
profile_builder → evidence_assessor → gap_finder
                                           │
                     (Send fan-out: one scout per target, in parallel)
                        ┌──────────────────┼──────────────────┐
                     scout              scout              scout
                        └──────────────────┼──────────────────┘
                                      junk_filter
                                           │
                                    audited_ranker
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and fill in a model key plus a GitHub token (a classic token
with zero scopes ticked — everything here is public data).

```bash
python -m app.main khushi-saxena
```

Or with the live UI, then open http://localhost:8000:

```bash
uvicorn app.server:app --reload --port 8000
```

To see it work with no keys and no network:

```bash
python demo.py
pytest -q
```

`demo.py` runs both products through the identical blocks; `pytest` runs 25
tests, all of them offline.

### Why `resume.txt` matters

Drop your resume in `resume.txt` (gitignored) and it gets folded into the
profile. This is not optional polish. Public repos systematically underrate
anyone whose serious work lives in private company repos:

| | GitHub only | GitHub + resume |
|---|---|---|
| Level | junior | mid |
| Strong skills | 1 | 5 |
| Growth targets | React, Django, Node.js | database internals, concurrency, distributed protocols |

Same account, same 17 repos, both runs.

## The six blocks

Each one is domain-agnostic. They live in `agentkit/blocks.py` and know nothing
about GitHub.

| Block | What it does |
|---|---|
| `ProfileBuilder` | Reads the subject's footprint and summarizes it |
| `EvidenceAssessor` | Extracts attributes — each must cite proof, enforced in Python |
| `GapFinder` | Picks growth targets, rejects ones already held at strong level |
| `ParallelSearcher` | Fans out one search per target, concurrently |
| `JunkFilter` | Quality-gates the feed with caller-supplied rules |
| `AuditedRanker` | Scores, then audits its own output and drops what it can't justify |

**Every block that produces judgement has something checking it.** The failure
mode is "returns nothing", not "returns something wrong". Without the audit
pass the system cheerfully recommended a Rust compiler issue to someone who has
never written Rust — the auditor now re-reads the attribute list and rejects
anything citing a skill that isn't there.

## Two executors, one set of blocks

```python
# plain Python, no framework required
from agentkit import Pipeline, make_llm
for name, result in Pipeline(blocks, domain, make_llm()).stream(state):
    print(name, result.summary)

# or the same blocks compiled to a LangGraph StateGraph
from agentkit.executors.langgraph_runner import build_graph, initial_state
final = build_graph(blocks, domain, make_llm()).invoke(initial_state("username"))
```

LangGraph buys `Send()` fan-out the graph waits on automatically, checkpointing,
and per-node streaming. `Pipeline` exists so adopting the SDK doesn't mean
adopting a framework. Blocks are written once.

## Writing an adapter

Four methods and a list of filter rules:

```python
class MyDomain:
    name = "my-domain"
    def fetch_subject(self, subject_id) -> dict: ...
    def search(self, target) -> list[dict]: ...
    def quality_signals(self, item) -> dict: ...
    def prompts(self) -> dict[str, str]: ...      # {} to use block defaults

def my_rule(item, signals):
    return "too old" if signals["age_days"] > 180 else None   # None = keep
```

Two ship with the repo. `adapters/github_oss.py` is the open-source matcher.
`adapters/conference_cfp.py` is a conference-talk matcher — a different product
on the identical blocks, in about 80 lines. `demo.py` runs both side by side.

## What broke, and what fixed it

The interesting part of this project is the debugging, not the architecture.

**Global issue search returns almost entirely spam.** Bounty farms,
Hacktoberfest label-spam, and one clone farm publishing the same issue text
under six repo names. The gatekeeper was correctly filtering 100% of results,
so the system returned nothing. The fix was not better filtering — it was
inverting discovery: find healthy repos first, then look inside them. Survival
went from 0% to ~100%, and runtime from 7 minutes to 91 seconds, because
`/repos` is not on the contended search quota.

**Stars are a weak quality filter.** The worst farm we hit had 242 of them. What
actually predicts a contribution landing is whether maintainers merge outside
PRs: that repo had 0 of its last 20 closed PRs merged, and 2,577 open issues.

**Secondary rate limits fire on bursts, not totals.** Parallel scouts tripped
them with 30/30 search quota remaining. Per-thread backoff made it worse — all
threads slept, woke together, and re-burst in lockstep. The fix is a cooldown
any thread can park and every thread respects, plus jitter on wake.

**`Node.js` is not a value GitHub's `language:` qualifier accepts.** It returns
zero results silently, which looks identical to a rate-limit failure. Hence
`LANGUAGE_ALIASES`.

**"Good first issue" labels are for docs and CSS.** Real database-internals work
never carries one, so requiring both a deep topic and a newcomer label searched
an intersection that barely exists. Deep targets now skip the label filter and
rank by whether the issue has real discussion and a substantial problem
statement instead.

`tests/test_github_ratelimit.py` has a test for each of these.

## Layout

```
agentkit/           the engine — knows no domain
  core.py           Block, Domain, Pipeline, state merging
  blocks.py         the six blocks
  llm.py            provider auto-detection (OpenAI or Anthropic)
  executors/        langgraph_runner.py — same blocks, compiled to a StateGraph
adapters/           the pluggable parts
  github_oss.py     GitHub API, rate limits, repo health, spam heuristics
  conference_cfp.py a second product, same blocks
app/                the running product
  pipeline.py       the block chain and its labels
  main.py           CLI
  server.py         FastAPI + SSE
  static/           the live UI
tests/              25 tests, all offline
```

## Known limits

- Recommendations can cluster in one or two large repos. `max_per_group` in the
  `JunkFilter` caps it, but the real lever is health-checking more repos per
  target.
- There is no held-out labelled set measuring whether the ranking is *good*,
  only tests proving the guard rails fire. That is the next thing worth
  building: 20 hand-labelled issues for a known profile, asserting the matcher
  agrees.
