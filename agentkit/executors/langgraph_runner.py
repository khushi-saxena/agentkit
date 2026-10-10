"""
LangGraph executor: runs the SAME blocks as agentkit.Pipeline, but as a
compiled StateGraph.

Why both executors exist. Pipeline is plain Python so the SDK is usable
without adopting a framework. LangGraph buys real things on top: Send() fan-out
that the graph waits on automatically, checkpointing, and per-node streaming.
Blocks are written once and run under either.

    build_graph([...blocks], domain, llm).invoke(initial_state)

The fan-out works by marking a block parallel=True with fan_out_over="targets";
this module turns that into a conditional edge returning one Send() per target.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..core import Block, BlockResult, Domain


class RunState(TypedDict, total=False):
    """
    Shared blackboard. The Annotated reducer on list keys is what lets several
    parallel scouts write to the same key without clobbering each other.
    """
    subject_id: str
    extra_context: str
    profile: dict
    attributes: list
    targets: list
    candidates: Annotated[list, operator.add]
    vetted: list
    ranked: list
    trace: Annotated[list, operator.add]


def build_graph(blocks: list[Block], domain: Domain,
                llm: Callable[[str], str], checkpointer=None):
    """Compile the blocks into a LangGraph StateGraph."""
    for b in blocks:
        b.bind(domain, llm)

    g = StateGraph(RunState)

    def wrap(block: Block):
        def node(state: dict) -> dict:
            result: BlockResult = block.run(state)
            updates = dict(result.updates)
            updates["trace"] = [{"block": block.name,
                                 "summary": result.summary,
                                 "facts": result.facts}]
            return updates
        return node

    for b in blocks:
        g.add_node(b.name, wrap(b))

    prev: str | None = None
    for b in blocks:
        if prev is None:
            g.add_edge(START, b.name)
        elif b.parallel and b.fan_out_over:
            key, target_node = b.fan_out_over, b.name

            def fan_out(state: dict, _key=key, _node=target_node):
                return [Send(_node, {**state, "_item": item})
                        for item in (state.get(_key) or [])]

            g.add_conditional_edges(prev, fan_out, [b.name])
        else:
            g.add_edge(prev, b.name)
        prev = b.name

    g.add_edge(prev, END)
    return g.compile(checkpointer=checkpointer)


def initial_state(subject_id: str, extra_context: str = "") -> dict:
    return {"subject_id": subject_id, "extra_context": extra_context,
            "profile": {}, "attributes": [], "targets": [],
            "candidates": [], "vetted": [], "ranked": [], "trace": []}
