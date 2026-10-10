"""
The same blocks must behave identically under the LangGraph executor, and the
SSE endpoint must stream one event per agent in order.
"""
import json

import pytest

from agentkit.executors.langgraph_runner import build_graph, initial_state
from adapters import github_oss
from tests.stubs import StubGitHub, stub_llm
from tests.test_blocks import blocks


def test_langgraph_runs_the_same_blocks_to_the_same_result():
    graph = build_graph(blocks(github_oss.RULES), StubGitHub(), stub_llm)
    final = graph.invoke(initial_state("khushi-saxena", "UBS, DRDO"))

    assert "Ghost Skill" not in [a["name"] for a in final["attributes"]]
    assert "Python" not in [t["target"] for t in final["targets"]]
    assert final["ranked"]
    allowed = {a["name"] for a in final["attributes"]}
    assert all(set(r["attributes_used"]) <= allowed for r in final["ranked"])


def test_langgraph_fans_out_one_scout_per_target():
    """Send() launches a scout per target and the graph waits for all of them."""
    graph = build_graph(blocks(github_oss.RULES), StubGitHub(), stub_llm)
    final = graph.invoke(initial_state("x", ""))

    targets = {t["target"] for t in final["targets"]}
    found_for = {c["found_for"] for c in final["candidates"]}
    assert found_for == targets


def test_trace_records_every_block_in_order():
    graph = build_graph(blocks(github_oss.RULES), StubGitHub(), stub_llm)
    final = graph.invoke(initial_state("x", ""))
    names = [t["block"] for t in final["trace"]]
    assert names[0] == "profile_builder"
    assert names[-1] == "audited_ranker"
    assert "junk_filter" in names


def test_sse_endpoint_streams_a_plan_then_every_agent(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    from app import server
    monkeypatch.setattr(server, "make_llm", lambda: stub_llm)
    monkeypatch.setattr(server, "build_domain", lambda: StubGitHub())

    client = fastapi_testclient.TestClient(server.app)
    events = []
    with client.stream("GET", "/api/run?username=khushi-saxena") as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            events.append(ev)
            if ev["type"] in ("finished", "error"):
                break

    assert events[0]["type"] == "plan"
    assert len(events[0]["agents"]) == 6

    errors = [e for e in events if e["type"] == "error"]
    assert not errors, errors

    done = [e["node"] for e in events if e["type"] == "agent_done"]
    assert done == ["profile_builder", "evidence_assessor", "gap_finder",
                    "parallel_searcher", "junk_filter", "audited_ranker"]

    matcher = [e for e in events if e.get("node") == "audited_ranker"][0]
    assert matcher["matches"]
    assert client.get("/").status_code == 200
