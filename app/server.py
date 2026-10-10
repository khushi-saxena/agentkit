"""
FastAPI + SSE server. Streams each agent's result to the browser as it
finishes, so you watch the pipeline fill in instead of staring at a terminal.

    uvicorn app.server:app --reload --port 8000

The mechanism: LangGraph's .stream(stream_mode="updates") yields one event per
node as it completes, carrying exactly what that node wrote to shared state. We
forward each one over Server-Sent Events, so the UI shows real state
transitions rather than a guessed progress bar.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI                                     # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse   # noqa: E402

from agentkit import make_llm                                   # noqa: E402
from agentkit.executors.langgraph_runner import (build_graph,   # noqa: E402
                                                 initial_state)
from app.pipeline import LABELS, ORDER, build_blocks, build_domain  # noqa: E402
from app.resume import load_resume                                 # noqa: E402

app = FastAPI()
HERE = os.path.dirname(__file__)


def _summarize(node: str, update: dict) -> dict:
    """Turn one node's raw state update into something worth showing a human."""
    trace = update.get("trace") or []
    base = trace[0] if trace else {"summary": "", "facts": []}
    out = {"detail": base.get("summary", ""), "facts": base.get("facts", [])}
    if node == "audited_ranker":
        out["matches"] = update.get("ranked", [])
    return out


def _run_into_queue(username: str, resume: str, q: "queue.Queue") -> None:
    try:
        graph = build_graph(build_blocks(), build_domain(), make_llm())
        for chunk in graph.stream(initial_state(username, resume),
                                  stream_mode="updates"):
            for node, update in chunk.items():
                label, desc = LABELS.get(node, (node, ""))
                q.put({"type": "agent_done", "node": node, "label": label,
                       "desc": desc, **_summarize(node, update)})
        q.put({"type": "finished"})
    except Exception as e:
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        q.put(None)


@app.get("/api/run")
async def run(username: str = "khushi-saxena"):
    try:
        resume, _ = load_resume()
    except RuntimeError as e:
        resume = ""
        print(f"[resume] {e}")

    async def events():
        q: "queue.Queue" = queue.Queue()
        threading.Thread(target=_run_into_queue,
                         args=(username, resume, q), daemon=True).start()
        # Send the whole plan first so the UI can render the steps greyed out.
        yield "data: " + json.dumps({
            "type": "plan",
            "agents": [{"node": n, "label": LABELS[n][0], "desc": LABELS[n][1]}
                       for n in ORDER],
            "has_resume": bool(resume),
        }) + "\n\n"
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield "data: " + json.dumps(item, default=str) + "\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))
