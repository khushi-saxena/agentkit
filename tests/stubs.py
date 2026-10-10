"""
Stub LLM and data sources so the whole suite runs offline, free, and fast.

The stub LLM deliberately returns BAD output in places - an attribute with no
evidence, a growth target the subject already has, a recommendation citing a
skill they lack - so the tests prove the guard rails actually fire rather than
just that the happy path runs.
"""
import json

ALLOWED = {"Python", "PostgreSQL", "Distributed Systems"}


def stub_llm(prompt: str) -> str:
    if "Summarize what this subject" in prompt:
        return json.dumps({
            "summary": "Builds full-stack AI tools and financial data backends.",
            "level": "mid", "tags": ["Python", "distributed systems"]})

    if "concrete attributes" in prompt:
        return json.dumps([
            {"name": "Python", "level": "strong", "evidence": "loglens, airslides"},
            {"name": "PostgreSQL", "level": "strong",
             "evidence": "Oracle to PostgreSQL migration at UBS"},
            {"name": "Distributed Systems", "level": "working",
             "evidence": "high-load financial data pipelines"},
            {"name": "Ghost Skill", "level": "strong", "evidence": ""},  # must be culled
        ])

    if "MOST DISTINCTIVE" in prompt or "speaking profile" in prompt:
        return json.dumps([
            {"target": "database internals", "why": "deepens the migration work",
             "builds_on": "PostgreSQL", "depth": "deep",
             "terms": ["storage", "engine"]},                 # list, not string
            {"target": "Python", "why": "sideways move", "builds_on": "Python",
             "depth": "shallow", "terms": "python"},          # must be rejected
        ])

    if "Score each candidate" in prompt:
        items = json.loads(prompt.split("CANDIDATES:")[1]
                           .split("For each candidate")[0].strip())
        return json.dumps([
            {"id": i["id"], "title": i["title"], "group": i["group"],
             "fit_score": 80 - n * 5, "stretch_score": 70,
             "reasoning": f"Cites the specific failure in {i['title'][:24]}.",
             "attributes_used": ["PostgreSQL"] if n == 0 else ["Rust"],
             "gained": "predicate pushdown"}
            for n, i in enumerate(items)])

    if "auditing another agent" in prompt:
        recs = json.loads(prompt.split("PROPOSED RECOMMENDATIONS:")[1]
                          .split("Reject any")[0].strip())
        return json.dumps([r for r in recs
                           if set(r["attributes_used"]) <= ALLOWED])

    return "{}"


def fenced_llm(prompt: str) -> str:
    """Same answers, wrapped in a markdown fence the way models often do."""
    return "```json\n" + stub_llm(prompt) + "\n```"


class StubGitHub:
    """GitHub adapter with the network swapped out."""
    name = "github-oss"

    def fetch_subject(self, subject_id):
        return {"repos": [{"name": "loglens", "language": "Python",
                           "description": "AI log analysis", "stars": 4}]}

    def search(self, target):
        t = target["target"]
        return [
            {"id": f"https://gh/{t}/0", "group": "farm/bounties",
             "title": "[ONBOARD: 2 RTC] Review an Open PR",
             "body": "x" * 400, "comments": 1, "repo_stars": 242,
             "updated_at": "2026-10-01T00:00:00Z"},
            {"id": f"https://gh/{t}/1", "group": "redis/redis",
             "title": "Replica crashes with SIGABRT after BGSAVE fork failure",
             "body": "x" * 600, "comments": 7, "repo_stars": 75000,
             "updated_at": "2026-10-05T00:00:00Z"},
            {"id": f"https://gh/{t}/2", "group": "redis/redis",
             "title": "Exact-count optimization folds NULL as false",
             "body": "x" * 600, "comments": 4, "repo_stars": 75000,
             "updated_at": "2026-10-06T00:00:00Z"},
            {"id": f"https://gh/{t}/3", "group": "redis/redis",
             "title": "Third issue from the same repo, should hit the cap",
             "body": "x" * 600, "comments": 3, "repo_stars": 75000,
             "updated_at": "2026-10-06T00:00:00Z"},
            {"id": f"https://gh/{t}/4", "group": "duckdb/duckdb",
             "title": "Tiny body issue", "body": "too short", "comments": 0,
             "repo_stars": 20000, "updated_at": "2026-10-06T00:00:00Z"},
        ]

    def quality_signals(self, item):
        return {"stars": item.get("repo_stars", 0),
                "comments": item.get("comments", 0),
                "body_len": len(item.get("body", "")),
                "age_days": 3}

    def prompts(self):
        return {}


CFPS = [
    {"name": "PyCon US", "track": "Databases", "url": "https://pycon/1",
     "topics": "storage engine internals python", "description": "d" * 300,
     "attendees": 3000, "days_to_deadline": 45},
    {"name": "SREcon", "track": "Storage", "url": "https://srecon/1",
     "topics": "storage engine reliability", "description": "d" * 300,
     "attendees": 1200, "days_to_deadline": 5},        # too soon
    {"name": "DeadConf", "track": "Any", "url": "https://dead/1",
     "topics": "storage engine", "description": "d" * 300,
     "attendees": 50, "days_to_deadline": -3},         # passed
]


def speaker_source(speaker_id):
    return {"past_talks": ["intro to FastAPI"], "blog_posts": 3}
