"""
agentkit core.

Nothing here knows about GitHub, conferences, or any specific domain. The
engine handles orchestration, parallel fan-out and state merging; a Domain
object supplies everything domain-specific.

  Block    - one agent. Reads shared state, returns the keys it updates.
  Domain   - the pluggable part: data source, quality signals, prompt overrides.
  Pipeline - runs blocks in order, fans out where asked, streams progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable


@runtime_checkable
class Domain(Protocol):
    """Implement these four methods and every block works for your domain."""

    name: str

    def fetch_subject(self, subject_id: str) -> dict:
        """The raw public footprint of whoever/whatever we're profiling."""

    def search(self, target: dict) -> list[dict]:
        """Run one search for a growth target, return candidate items."""

    def quality_signals(self, item: dict) -> dict:
        """Signals a filter can judge: stars, age, activity, length."""

    def prompts(self) -> dict[str, str]:
        """Prompt overrides keyed by block name. {} to use block defaults."""


@dataclass
class BlockResult:
    """State updates plus something a human can read."""
    updates: dict[str, Any]
    summary: str = ""
    facts: list[str] = field(default_factory=list)


class Block:
    """
    Base class for every agent.

    A block never talks to a specific API. It goes through self.domain and
    self.llm, both injected at bind time. That indirection is the whole reason
    the same six blocks power two different products.
    """

    name: str = "block"
    #: run one copy per item in the state key named by fan_out_over
    parallel: bool = False
    fan_out_over: str | None = None

    def __init__(self, **config):
        self.config = config
        self.domain: Domain | None = None
        self.llm: Callable[[str], str] | None = None

    def bind(self, domain: Domain, llm: Callable[[str], str]) -> "Block":
        self.domain, self.llm = domain, llm
        return self

    def prompt(self, default: str, **kw) -> str:
        """Domain's override for this block if it has one, else the default."""
        overrides = self.domain.prompts() if self.domain else {}
        return overrides.get(self.name, default).format(**kw)

    def run(self, state: dict) -> BlockResult:
        raise NotImplementedError


def merge(state: dict, updates: dict) -> dict:
    """Lists concatenate so parallel blocks accumulate; everything else replaces."""
    out = dict(state)
    for k, v in updates.items():
        if isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


@dataclass
class Pipeline:
    """
    The simple executor: plain Python, no framework required.

    Deliberately not tied to LangGraph. An SDK that forces a framework on you
    is a harder sell, and LangGraph is available as an alternative executor
    (agentkit.executors.langgraph_runner) over these exact same blocks.
    """
    blocks: list[Block]
    domain: Domain
    llm: Callable[[str], str]
    max_workers: int = 4

    def __post_init__(self):
        for b in self.blocks:
            b.bind(self.domain, self.llm)

    def stream(self, initial: dict) -> Iterator[tuple[str, BlockResult]]:
        """Run, yielding (block_name, result) as each block finishes."""
        state = dict(initial)
        for block in self.blocks:
            if block.parallel and block.fan_out_over:
                result = self._fan_out(block, state)
            else:
                result = block.run(state)
            state = merge(state, result.updates)
            yield block.name, result

    def run(self, initial: dict) -> dict:
        state = dict(initial)
        for _, result in self.stream(initial):
            state = merge(state, result.updates)
        return state

    def _fan_out(self, block: Block, state: dict) -> BlockResult:
        from concurrent.futures import ThreadPoolExecutor

        items = state.get(block.fan_out_over) or []
        if not items:
            return BlockResult({}, "nothing to fan out over")

        def one(item):
            sub = dict(state)
            sub["_item"] = item
            return block.run(sub)

        merged: dict[str, Any] = {}
        facts: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for res in pool.map(one, items):
                merged = merge(merged, res.updates)
                facts.extend(res.facts)
        return BlockResult(merged, f"{len(items)} ran in parallel", facts)
