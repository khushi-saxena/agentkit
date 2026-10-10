"""
Rate-limit tests for the GitHub adapter.

These cover the three things that took the longest to get right:

1. A primary (quota) 403 and a secondary (burst) 403 need different responses.
   Conflating them produced a misleading "set GITHUB_TOKEN" message while the
   token was fine with 4999/5000 remaining.

2. A secondary limit must retry with backoff, honouring Retry-After.

3. Backoff must be SHARED. Per-thread backoff made things worse: all threads
   slept, woke together and re-burst in lockstep.
"""
import threading
import time

import pytest

from adapters import github_oss as gh


class FakeResponse:
    def __init__(self, status=200, text="", headers=None, payload=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"items": []}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def reset_throttle(monkeypatch):
    monkeypatch.setattr(gh, "_last_search_at", 0.0, raising=False)
    monkeypatch.setattr(gh, "_cooldown_until", 0.0, raising=False)
    monkeypatch.setattr(gh, "MIN_SEARCH_GAP", 1.0)
    yield


def test_primary_limit_raises_immediately_without_retrying(monkeypatch):
    calls = []

    def one_shot(url, **kw):
        calls.append(url)
        return FakeResponse(403, "API rate limit exceeded",
                            {"X-RateLimit-Remaining": "0",
                             "X-RateLimit-Reset": str(int(time.time()) + 1800)})

    monkeypatch.setattr(gh.requests, "get", one_shot)
    with pytest.raises(RuntimeError) as err:
        gh._get("/users/x/repos")

    assert "PRIMARY" in str(err.value)
    assert "min" in str(err.value)          # tells you when it resets
    assert len(calls) == 1, "primary limit must not retry - waiting cannot help"


def test_secondary_limit_retries_and_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def flaky(url, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResponse(403, "You have exceeded a secondary rate limit.",
                                {"Retry-After": "1"})
        return FakeResponse(200, payload={"items": [{"full_name": "redis/redis"}]})

    monkeypatch.setattr(gh.requests, "get", flaky)
    out = gh._get("/search/repositories", {"q": "storage"})

    assert attempts["n"] == 2
    assert out["items"][0]["full_name"] == "redis/redis"


def test_one_thread_hitting_the_limit_pauses_every_thread(monkeypatch):
    """The lockstep fix: a parked cooldown is global, not per-thread."""
    lock = threading.Lock()
    stamps, first = [], {"done": False}

    def handler(url, **kw):
        with lock:
            stamps.append(time.monotonic())
            if not first["done"]:
                first["done"] = True
                return FakeResponse(403, "You have exceeded a secondary rate limit.",
                                    {"Retry-After": "3"})
        return FakeResponse(200)

    monkeypatch.setattr(gh.requests, "get", handler)

    start = time.monotonic()
    threads = [threading.Thread(target=gh._get,
                                args=("/search/repositories", {"q": "x"}))
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start

    assert elapsed >= 3.0, f"cooldown was not shared across threads ({elapsed:.1f}s)"
    gaps = [b - a for a, b in zip(sorted(stamps), sorted(stamps)[1:])]
    assert max(gaps) >= 2.5, f"no shared pause visible in gaps {gaps}"


def test_parallel_searches_are_staggered(monkeypatch):
    stamps = []
    monkeypatch.setattr(gh.requests, "get",
                        lambda url, **kw: (stamps.append(time.monotonic()),
                                           FakeResponse(200))[1])

    threads = [threading.Thread(target=gh._get,
                                args=("/search/repositories", {"q": "x"}))
               for _ in range(3)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert time.monotonic() - start >= 2.0, "searches fired as a burst"


def test_repo_lookups_are_not_throttled(monkeypatch):
    """/repos is off the contended search quota - that is why repo-first is fast."""
    monkeypatch.setattr(gh.requests, "get",
                        lambda url, **kw: FakeResponse(200, payload={"ok": True}))
    start = time.monotonic()
    for _ in range(5):
        gh._get("/repos/redis/redis")
    assert time.monotonic() - start < 1.0


# --- the quieter bugs ------------------------------------------------------

def test_language_aliases_map_onto_real_github_names():
    """'Node.js' is not a value GitHub's language: qualifier accepts."""
    assert gh.normalize_language("Node.js") == "JavaScript"
    assert gh.normalize_language("Django") == "Python"
    assert gh.normalize_language("SwiftUI") == "Swift"
    assert gh.normalize_language("Python") == "Python"
    assert gh.normalize_language("Zig") == "Zig"        # unknown passes through


def test_junk_markers_catch_the_real_farm_titles():
    """Verbatim titles from a run where all four recommendations were junk."""
    junk = [
        "[ONBOARD: 2 RTC] Review an Open PR",
        "🍒 Call for Contributors — Build the future of local LLMs!",
        "[BOUNTY] Star the repo and share why (0.5 RTC)",
        "[Hacktoberfest] Add APort to an \"Awesome\" List",
    ]
    real = [
        "Replica crashes with SIGABRT after BGSAVE fork failure",
        "Add type hints to utils/parser.py",
        "Positional arguments passed as kwargs confuse some decorators",
    ]
    for title in junk:
        assert gh.no_farm_titles({"title": title}, {}) is not None, title
    for title in real:
        assert gh.no_farm_titles({"title": title}, {}) is None, title


def test_health_check_rejects_a_farm_and_keeps_a_real_repo(monkeypatch):
    """Stars alone never worked - the farm had 242. Merge activity is the signal."""
    def fake(url, **kw):
        if "/pulls" in url:
            if "farm" in url:
                return FakeResponse(200, payload=[{"merged_at": None}] * 20)
            return FakeResponse(200, payload=[{"merged_at": "2026-10-01"}] * 12)
        if "farm" in url:
            return FakeResponse(200, payload={"stargazers_count": 242,
                                              "archived": False,
                                              "open_issues_count": 2577})
        return FakeResponse(200, payload={"stargazers_count": 75000,
                                          "archived": False,
                                          "open_issues_count": 180})

    monkeypatch.setattr(gh.requests, "get", fake)
    domain = gh.GitHubOSSDomain()

    farm = domain.check_repo_health("farm/bounties")
    good = domain.check_repo_health("redis/redis")

    assert not farm["ok"]
    assert "merged" in farm["reason"]
    assert good["ok"] and good["stars"] == 75000


def test_deep_targets_skip_the_newcomer_label(monkeypatch):
    """
    Real internals work is never labelled "good first issue". Requiring both
    searched an intersection that barely exists.
    """
    seen = {}

    def capture(url, params=None, **kw):
        seen.update(params or {})
        return FakeResponse(200, payload=[])

    monkeypatch.setattr(gh.requests, "get", capture)
    domain = gh.GitHubOSSDomain()

    domain.list_repo_issues("redis/redis", deep=True)
    assert "labels" not in seen

    seen.clear()
    domain.list_repo_issues("redis/redis", deep=False)
    assert seen.get("labels") == "good first issue"


# --- query construction ----------------------------------------------------

def test_query_ladder_broadens_a_human_phrase():
    """
    The planner writes for humans. GitHub repo search ANDs every word against
    name/description/README, so a comma-separated phrase matches nothing.
    """
    ladder = gh.GitHubOSSDomain.query_ladder(
        "PostgreSQL internals, database optimization, query planning")

    assert "," not in " ".join(ladder)
    assert ladder[0].count(" ") <= 1, "first rung should already be short"
    assert ladder[-1].startswith("topic:"), "broadest rung should be a topic"
    assert any("PostgreSQL" in q for q in ladder)


def test_query_ladder_strips_generic_filler():
    ladder = gh.GitHubOSSDomain.query_ladder("Advanced TypeScript Patterns")
    assert "TypeScript" in ladder
    assert "topic:typescript" in ladder


def test_find_repos_walks_the_ladder_until_something_matches(monkeypatch):
    tried = []

    def only_broad_query_works(url, params=None, **kw):
        q = (params or {}).get("q", "")
        tried.append(q)
        if q.startswith("topic:"):
            return FakeResponse(200, payload={
                "items": [{"full_name": "postgres/postgres",
                           "stargazers_count": 17000}]})
        return FakeResponse(200, payload={"items": []})

    monkeypatch.setattr(gh.requests, "get", only_broad_query_works)
    repos = gh.GitHubOSSDomain().find_repos(
        "PostgreSQL internals, database optimization, query planning")

    assert len(tried) == 3, f"should have tried every rung: {tried}"
    assert repos[0]["full_name"] == "postgres/postgres"


def test_find_repos_stops_at_the_first_rung_that_works(monkeypatch):
    """No point broadening further once we have results."""
    tried = []

    def always_works(url, params=None, **kw):
        tried.append((params or {}).get("q", ""))
        return FakeResponse(200, payload={
            "items": [{"full_name": "duckdb/duckdb", "stargazers_count": 20000}]})

    monkeypatch.setattr(gh.requests, "get", always_works)
    gh.GitHubOSSDomain().find_repos("database internals")
    assert len(tried) == 1, f"broadened unnecessarily: {tried}"


# --- transient faults ------------------------------------------------------

def test_connection_drop_is_retried(monkeypatch):
    """A dropped connection killed an entire scout in one run."""
    attempts = {"n": 0}

    def flaky(url, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise gh.requests.ConnectionError("Remote end closed connection")
        return FakeResponse(200, payload={"ok": True})

    monkeypatch.setattr(gh.requests, "get", flaky)
    monkeypatch.setattr(gh.time, "sleep", lambda s: None)
    assert gh._get("/repos/redis/redis") == {"ok": True}
    assert attempts["n"] == 2


def test_server_errors_are_retried(monkeypatch):
    attempts = {"n": 0}

    def flaky(url, **kw):
        attempts["n"] += 1
        return (FakeResponse(502) if attempts["n"] == 1
                else FakeResponse(200, payload={"ok": True}))

    monkeypatch.setattr(gh.requests, "get", flaky)
    monkeypatch.setattr(gh.time, "sleep", lambda s: None)
    assert gh._get("/repos/x/y") == {"ok": True}


def test_unreachable_host_eventually_raises_clearly(monkeypatch):
    def always_down(url, **kw):
        raise gh.requests.ConnectionError("no route to host")

    monkeypatch.setattr(gh.requests, "get", always_down)
    monkeypatch.setattr(gh.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="unreachable"):
        gh._get("/repos/x/y")


# --- title and relevance gates ---------------------------------------------

def test_cjk_titles_are_not_treated_as_vague():
    """
    Counting whitespace-separated words rejected every Chinese, Japanese and
    Korean title, including specific ones.
    """
    specific_cjk = "[Bug] 添加代理后还是无法解析tiktok地址"
    assert gh.substantive_title({"title": specific_cjk}, {}) is None
    assert gh.substantive_title(
        {"title": "ビルドが失敗する際のエラーメッセージが不明確"}, {}) is None
    # still rejects genuinely contentless CJK titles
    assert gh.substantive_title({"title": "バグ"}, {})


def test_relevance_gate_drops_drifted_candidates():
    """
    Search broadens when a narrow query returns nothing, so results drift. A
    run aimed at the Python C API returned GPU bugs in a Stable Diffusion UI.
    """
    drifted = {"title": "KSampler Doesnt Run",
               "body": "CUDA error: HIPBLAS_STATUS_INVALID_VALUE",
               "group": "Comfy-Org/ComfyUI",
               "target_terms": "Python C API", "labels": []}
    assert gh.relevant_to_target(drifted, {}) is not None

    on_target = {"title": "Prisma 8 db update bug",
                 "body": "migration fails on the postgres database",
                 "group": "prisma/orm",
                 "target_terms": "database internals", "labels": []}
    assert gh.relevant_to_target(on_target, {}) is None


def test_relevance_gate_ignores_generic_terms():
    """'advanced development patterns' has no checkable signal - don't drop on it."""
    item = {"title": "anything", "body": "", "group": "o/r",
            "target_terms": "advanced development patterns", "labels": []}
    assert gh.relevant_to_target(item, {}) is None
