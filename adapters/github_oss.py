"""
Adapter: open-source contribution matcher.

Everything GitHub-specific lives here - API calls, rate-limit handling, repo
health, spam heuristics. Nothing here knows about orchestration, evidence
enforcement or the audit pass; those are agentkit's.

Two things in this file were learned the hard way and are worth reading:

1. Discovery is repo-first. Global issue search returns almost entirely spam
   (bounty farms, Hacktoberfest label-spam, clone farms publishing the same
   issue under six repo names), so the filter culled ~100% and nothing
   survived. Finding healthy repos first and then listing their issues took
   survival from 0% to ~100% and cut a run from 7 minutes to 91 seconds,
   because /repos is not on the contended search quota.

2. Secondary rate limits fire on BURSTS, not totals. Parallel searchers tripped
   them with 30/30 search quota remaining. Per-thread backoff made it worse:
   all threads slept, woke together and re-burst. The fix is a cooldown that
   any thread can park and every thread respects.
"""
from __future__ import annotations

import os
import random
import re
import threading
import time

import requests

API = "https://api.github.com"
MAX_RETRIES = 4
MIN_SEARCH_GAP = 10.0          # seconds between any two search requests
REQUEST_TIMEOUT = 20

_search_lock = threading.Lock()
_last_search_at = 0.0
_cooldown_until = 0.0          # parked by ANY thread, respected by ALL


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "agentkit"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _reserve_search_slot() -> None:
    """
    Claim the next search slot. The wait is computed under the lock so two
    threads cannot both decide they are clear, and the sleep happens outside it
    with jitter so woken threads do not re-burst in lockstep.
    """
    global _last_search_at
    while True:
        with _search_lock:
            now = time.monotonic()
            earliest = max(_last_search_at + MIN_SEARCH_GAP, _cooldown_until)
            if now >= earliest:
                _last_search_at = now
                return
            wait = earliest - now
        time.sleep(wait + random.uniform(0, 1.5))


def _park_cooldown(seconds: float) -> None:
    """One thread hit a burst limit - stop every thread from searching."""
    global _cooldown_until
    with _search_lock:
        _cooldown_until = max(_cooldown_until, time.monotonic() + seconds)


def _classify_403(resp) -> str:
    """
    GitHub's 403s need different responses. Primary = quota gone, waiting
    inside one run will not help. Secondary = burst detection, transient.
    """
    body = resp.text.lower()
    if "secondary rate limit" in body:
        return "secondary"
    if resp.headers.get("X-RateLimit-Remaining") == "0":
        return "primary"
    if "rate limit" in body:
        return "secondary"
    return "other"


def _get(path: str, params: dict | None = None, _attempt: int = 1):
    is_search = path.startswith("/search")
    if is_search:
        _reserve_search_slot()

    try:
        r = requests.get(f"{API}{path}", headers=_headers(),
                         params=params, timeout=REQUEST_TIMEOUT)
    except (requests.ConnectionError, requests.Timeout) as e:
        # A dropped connection killed a whole scout in one run. Transient
        # network faults deserve the same treatment as a burst limit.
        if _attempt > MAX_RETRIES:
            raise RuntimeError(f"GitHub unreachable after {MAX_RETRIES} "
                               f"retries: {e}") from e
        wait = min(30, 2 ** _attempt)
        print(f"    [github] connection dropped - retrying in {wait}s "
              f"({_attempt}/{MAX_RETRIES})")
        time.sleep(wait)
        return _get(path, params, _attempt + 1)

    if r.status_code in (500, 502, 503, 504) and _attempt <= MAX_RETRIES:
        wait = min(30, 2 ** _attempt)
        print(f"    [github] HTTP {r.status_code} - retrying in {wait}s "
              f"({_attempt}/{MAX_RETRIES})")
        time.sleep(wait)
        return _get(path, params, _attempt + 1)

    if r.status_code in (403, 429):
        kind = _classify_403(r)

        if kind == "primary":
            reset = r.headers.get("X-RateLimit-Reset", "")
            when = ""
            if reset.isdigit():
                when = f" Resets in ~{max(0, int((int(reset) - time.time()) / 60))} min."
            hint = ("" if os.environ.get("GITHUB_TOKEN") else
                    " No GITHUB_TOKEN set - with one you get 5000 req/hour instead of 60.")
            raise RuntimeError("GitHub PRIMARY rate limit exhausted." + when + hint)

        if kind == "secondary" and _attempt <= MAX_RETRIES:
            retry_after = r.headers.get("Retry-After", "")
            wait = (float(retry_after) if retry_after.isdigit()
                    else min(60, 5 * 2 ** (_attempt - 1)))
            print(f"    [github] burst limit - all searches paused {wait:.0f}s "
                  f"(retry {_attempt}/{MAX_RETRIES})")
            if is_search:
                _park_cooldown(wait)
            time.sleep(wait)
            return _get(path, params, _attempt + 1)

        if kind == "secondary":
            raise RuntimeError(
                f"GitHub burst limit still firing after {MAX_RETRIES} retries. "
                "This is abuse detection, not your quota - your token is fine. "
                "Wait a few minutes, or raise MIN_SEARCH_GAP.")

    r.raise_for_status()
    return r.json()


# GitHub's language: qualifier only accepts its own linguist names. Growth
# targets are human phrases, and an unmapped one silently matches nothing -
# which looks exactly like a rate-limit failure but isn't.
LANGUAGE_ALIASES = {
    "node.js": "JavaScript", "node": "JavaScript", "nodejs": "JavaScript",
    "express": "JavaScript", "react": "JavaScript", "vue": "JavaScript",
    "next.js": "TypeScript", "angular": "TypeScript", "ts": "TypeScript",
    "js": "JavaScript", "swiftui": "Swift", "uikit": "Swift",
    "django": "Python", "flask": "Python", "fastapi": "Python",
    "pytorch": "Python", "asyncio": "Python", "async python": "Python",
    "pytest": "Python", "langchain": "Python", "langgraph": "Python",
    "spring": "Java", "spring boot": "Java", "rails": "Ruby",
    ".net": "C#", "dotnet": "C#", "golang": "Go", "c++": "CPP",
}

JUNK_MARKERS = (
    "onboard", "bounty", "hacktoberfest", "call for contributor",
    "call for maintainer", "airdrop", "giveaway", "claim your", "earn ",
    "reward", "introduce yourself", "add your name", "star the repo",
)


def normalize_language(name: str) -> str:
    """Map a human skill name onto something `language:` will actually match."""
    key = (name or "").strip().lower()
    return LANGUAGE_ALIASES.get(key, (name or "").strip())


class GitHubOSSDomain:
    """Profiles a developer's GitHub and finds issues worth contributing to."""

    name = "github-oss"

    def __init__(self, min_stars: int = 500, repos_per_target: int = 10,
                 health_checks: int = 5, issues_per_repo: int = 3):
        self.min_stars = min_stars
        self.repos_per_target = repos_per_target
        self.health_checks = health_checks
        self.issues_per_repo = issues_per_repo

    # ---- what the subject has built -------------------------------------
    def fetch_subject(self, subject_id: str) -> dict:
        repos = _get(f"/users/{subject_id}/repos",
                     {"per_page": 20, "sort": "updated", "type": "owner"})
        return {"repos": [
            {"name": r["name"], "language": r.get("language") or "unknown",
             "description": (r.get("description") or "")[:120],
             "stars": r.get("stargazers_count", 0)}
            for r in repos if not r.get("fork")]}

    # ---- repo-first discovery -------------------------------------------
    @staticmethod
    def query_ladder(terms: str) -> list[str]:
        """
        Progressively broader free-text queries, broadest last.

        GitHub's repository search ANDs every term and only matches against
        name, description and README - so a growth target phrased as
        "PostgreSQL internals, database optimization, query planning" matches
        literally nothing, commas included. The planner writes for humans; this
        turns that into something the search API can actually satisfy.
        """
        cleaned = re.sub(r"[,;/|]+", " ", terms or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        words = [w for w in cleaned.split() if len(w) > 2]

        # Drop generic filler so the surviving words carry real signal.
        filler = {"advanced", "development", "programming", "modern", "deep",
                  "internals", "patterns", "systems", "engineering", "design",
                  "and", "the", "for", "with", "using"}
        distinctive = [w for w in words if w.lower() not in filler]

        ladder: list[str] = []
        if len(words) >= 2:
            ladder.append(" ".join(words[:2]))
        if distinctive:
            ladder.append(distinctive[0])
            ladder.append(f"topic:{distinctive[0].lower()}")
        elif words:
            ladder.append(words[0])
            ladder.append(f"topic:{words[0].lower()}")

        seen, out = set(), []
        for q in ladder:
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def find_repos(self, terms: str, language: str = "") -> list[dict]:
        """Walk the ladder, stopping at the first query that returns anything."""
        recent = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 60 * 86400))
        lang = f"language:{normalize_language(language)}" if language else ""

        for query in self.query_ladder(terms):
            parts = [query, f"stars:>={self.min_stars}", "archived:false",
                     f"pushed:>={recent}", lang]
            data = _get("/search/repositories",
                        {"q": " ".join(p for p in parts if p),
                         "sort": "stars", "per_page": self.repos_per_target})
            items = data.get("items", [])
            if items:
                if query != terms:
                    print(f"    [github] broadened {terms!r} -> {query!r}")
                return items
        return []

    def check_repo_health(self, full_name: str) -> dict:
        """
        Stars are a weak signal - a bounty farm we hit had 242 of them. What
        actually predicts a contribution landing is whether maintainers merge
        outside PRs, recently.
        """
        try:
            repo = _get(f"/repos/{full_name}")
        except Exception as e:
            return {"ok": False, "reason": f"lookup failed: {e}", "stars": 0,
                    "merged_recently": 0}

        stars = repo.get("stargazers_count", 0)
        open_issues = repo.get("open_issues_count", 0)
        merged = 0
        try:
            prs = _get(f"/repos/{full_name}/pulls",
                       {"state": "closed", "per_page": 20,
                        "sort": "updated", "direction": "desc"})
            merged = sum(1 for p in prs if p.get("merged_at"))
        except Exception:
            pass

        reasons = []
        if repo.get("archived"):
            reasons.append("archived")
        if stars < self.min_stars // 5:
            reasons.append(f"only {stars} stars")
        if merged < 3:
            reasons.append(f"only {merged}/20 recent PRs merged")
        if open_issues > 2000 and merged < 5:
            reasons.append(f"{open_issues} open issues but few merges (issue farm)")

        return {"ok": not reasons, "reason": "; ".join(reasons) or "healthy",
                "stars": stars, "merged_recently": merged}

    def list_repo_issues(self, full_name: str, deep: bool) -> list[dict]:
        """
        Deep targets skip the newcomer-label filter entirely.

        Real database-internals or concurrency work is never labelled "good
        first issue" - those labels are for docs and CSS - so requiring both
        searched an intersection that barely exists.
        """
        params = {"state": "open", "assignee": "none", "per_page": 20}
        if not deep:
            params["labels"] = "good first issue"

        try:
            issues = _get(f"/repos/{full_name}/issues", params)
        except Exception:
            return []

        out = []
        for i in issues:
            if "pull_request" in i:
                continue
            out.append({
                "id": i["html_url"], "group": full_name, "title": i["title"],
                "body": (i.get("body") or "").strip()[:1200],
                "comments": i.get("comments", 0),
                "labels": [l["name"] for l in i.get("labels", [])],
                "updated_at": i.get("updated_at", ""),
            })

        if deep:
            # No labels to rank by, so rank by signals of a real problem:
            # someone described it, and the statement is substantial.
            out.sort(key=lambda i: (i["comments"] >= 2, len(i["body"])), reverse=True)
        return out[: self.issues_per_repo]

    def search(self, target: dict) -> list[dict]:
        """
        Repo-first discovery, with the funnel printed at every stage.

        An empty result has four possible causes and they need different fixes,
        so "0 candidates" on its own is useless. This prints repos found, which
        passed health and why the rest failed, and issues per repo.
        """
        deep = target.get("depth", "deep") == "deep"
        terms = target.get("terms") or target.get("target", "")
        label = target.get("target", terms)

        repos = self.find_repos(terms, target.get("language", ""))
        print(f"    [{label}] query={terms!r} -> {len(repos)} repos")
        if not repos:
            print(f"    [{label}] no repos matched. Terms may be too specific, "
                  f"or min_stars={self.min_stars} too high.")
            return []

        found: list[dict] = []
        for repo in repos[: self.health_checks]:
            health = self.check_repo_health(repo["full_name"])
            if not health["ok"]:
                print(f"    [{label}]   x {repo['full_name']}: {health['reason']}")
                continue
            issues = self.list_repo_issues(repo["full_name"], deep)
            print(f"    [{label}]   + {repo['full_name']} "
                  f"({health['stars']} stars, {health['merged_recently']}/20 merged) "
                  f"-> {len(issues)} open issues")
            for issue in issues:
                issue["repo_stars"] = health["stars"]
                issue["repo_merged_recently"] = health["merged_recently"]
                issue["target_terms"] = terms
                found.append(issue)

        if not found:
            hint = ("deep path: unassigned open issues were empty"
                    if deep else
                    "shallow path: no issue carried a 'good first issue' label")
            print(f"    [{label}] repos were healthy but yielded nothing ({hint}).")
        return found

    # ---- what the filter judges on ---------------------------------------
    def quality_signals(self, item: dict) -> dict:
        age_days = 9999
        if item.get("updated_at"):
            try:
                t = time.strptime(item["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
                age_days = (time.time() - time.mktime(t)) / 86400
            except ValueError:
                pass
        return {"stars": item.get("repo_stars", 0),
                "merged_recently": item.get("repo_merged_recently", 0),
                "comments": item.get("comments", 0),
                "body_len": len(item.get("body", "")),
                "age_days": round(age_days)}

    def prompts(self) -> dict[str, str]:
        return {}            # the generic block prompts already fit this domain


# ---- filter rules: return a reason to drop, or None to keep ---------------
def no_farm_titles(item, signals):
    t = item.get("title", "").lower()
    hit = next((m for m in JUNK_MARKERS if m in t), None)
    return f"farm marker '{hit.strip()}'" if hit else None


def real_problem_statement(item, signals):
    return "body too short" if signals["body_len"] < 200 else None


def not_stale(item, signals):
    return "no activity in 180+ days" if signals["age_days"] > 180 else None


# CJK, Thai and similar scripts do not delimit words with spaces, so counting
# whitespace-separated words drops every Chinese, Japanese and Korean title.
_DENSE_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u0e00-\u0e7f]")


def substantive_title(item, signals):
    """
    A run surfaced an issue literally titled "farm". A title you cannot act on
    is not a recommendation, whatever the body says.

    Scripts without word spacing are measured by character count instead -
    counting spaces would reject every CJK title, including specific ones.
    """
    title = (item.get("title") or "").strip()
    if not title:
        return "no title"

    stripped = re.sub(r"^\s*\[[^\]]{0,20}\]\s*", "", title)  # drop a [Bug] prefix

    if _DENSE_SCRIPT.search(stripped):
        dense = len(_DENSE_SCRIPT.findall(stripped))
        if dense < 6 and len(stripped) < 12:
            return f"title too vague to act on ({title!r})"
        return None

    words = [w for w in re.split(r"[\s\-_/]+", stripped) if w]
    if len(words) < 3 or len(stripped) < 12:
        return f"title too vague to act on ({title!r})"
    return None


STOPWORDS = {"advanced", "development", "programming", "modern", "deep",
             "internals", "patterns", "systems", "engineering", "design",
             "api", "and", "the", "for", "with", "using", "data"}


def relevant_to_target(item, signals):
    """
    The candidate must actually mention the growth area that found it.

    Search broadens when a narrow query returns nothing ("Python C API" ends up
    as "Python"), and the results then drift: a run aimed at database internals
    and the Python C API returned GPU bugs in a Stable Diffusion UI. Asking the
    model to self-report relevance did not work - it marked them on-target
    because they "involve Python". So this is checked in code.
    """
    terms = (item.get("target_terms") or "").lower()
    keys = [t for t in re.split(r"[^a-z0-9+#.]+", terms)
            if len(t) > 2 and t not in STOPWORDS]
    if not keys:
        return None                      # nothing specific to check against

    haystack = " ".join([
        item.get("title") or "", item.get("body") or "",
        item.get("group") or "", " ".join(item.get("labels") or []),
    ]).lower()

    if not any(k in haystack for k in keys):
        return f"never mentions {'/'.join(keys[:3])} (drifted off-target)"
    return None


RULES = [no_farm_titles, real_problem_statement, not_stale, substantive_title,
         relevant_to_target]
