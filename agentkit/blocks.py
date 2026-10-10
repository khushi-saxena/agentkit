"""
The six reusable blocks. None of them mention GitHub, conferences, or any
other domain.

  ProfileBuilder   - read a subject's footprint, summarize it
  EvidenceAssessor - extract attributes, each requiring cited proof
  GapFinder        - pick growth targets, reject ones already held
  ParallelSearcher - fan out one search per target
  JunkFilter       - quality-gate the candidate feed
  AuditedRanker    - score, then audit its own output

Every block that produces judgement has something checking it. That is the
design principle: the failure mode is "returns nothing", not "returns
something wrong".
"""
from __future__ import annotations

import json

from .core import Block, BlockResult


def parse_json(raw: str):
    """Models fence their JSON about half the time. Strip it either way."""
    raw = raw.strip()
    for fence in ("```json", "```"):
        raw = raw.removeprefix(fence)
    return json.loads(raw.removesuffix("```").strip())


def as_text(value) -> str:
    """Growth-target keywords come back as a list from some providers."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value or "")


class ProfileBuilder(Block):
    name = "profile_builder"

    DEFAULT = """Here is raw information about a subject.

{raw}

ADDITIONAL CONTEXT:
{extra}

Summarize what this subject actually is or does. Reply with ONLY JSON:
{{"summary": "two sentences",
  "level": "one word - pick from: beginner, junior, mid, senior",
  "tags": ["3-6 concrete technologies or domains"]}}

Do not echo the descriptions above as values. Fill them in."""

    def run(self, state: dict) -> BlockResult:
        raw = self.domain.fetch_subject(state["subject_id"])
        profile = parse_json(self.llm(self.prompt(
            self.DEFAULT,
            raw=json.dumps(raw, default=str)[:4000],
            extra=(state.get("extra_context") or "(none supplied)")[:3000],
        )))
        profile["raw"] = raw
        return BlockResult(
            {"profile": profile},
            profile.get("summary", ""),
            [f"level: {profile.get('level')}",
             "tags: " + ", ".join(profile.get("tags", [])[:5])],
        )


class EvidenceAssessor(Block):
    """
    Attributes must cite evidence. The rule is enforced in Python below, not
    just requested in the prompt - asking a model nicely is not a guarantee.
    """
    name = "evidence_assessor"

    DEFAULT = """Assess this subject's concrete attributes.

PROFILE: {summary}
RAW: {raw}
ADDITIONAL CONTEXT: {extra}

Rules:
- List {count} attributes.
- EVERY attribute must cite specific evidence from the material above.
- If you cannot cite evidence, omit the attribute entirely.
- level is one of: strong, working, beginner.
- Each attribute must be a CONCRETE technology, technique or domain that
  someone could put on a CV: "PostgreSQL", "distributed systems",
  "multithreaded C++". Never a meta-observation about the subject or their
  profile - things like "Project Visibility", "Diverse Programming Languages"
  or "Open Source Contribution" are not skills and must be omitted.
- Do not pad the list. Six real attributes beat ten with filler.

Reply with ONLY a JSON array:
[{{"name": "...", "level": "...", "evidence": "..."}}]"""

    def run(self, state: dict) -> BlockResult:
        p = state["profile"]
        attrs = parse_json(self.llm(self.prompt(
            self.DEFAULT,
            summary=p.get("summary", ""),
            raw=json.dumps(p.get("raw"), default=str)[:3000],
            extra=(state.get("extra_context") or "(none supplied)")[:2500],
            count=self.config.get("count", "6-10"),
        )))
        kept = [a for a in attrs if str(a.get("evidence", "")).strip()]
        culled = len(attrs) - len(kept)
        strong = [a for a in kept if a.get("level") == "strong"]
        facts = [f"{a['name']} ({a['level']}) - {a['evidence'][:60]}" for a in kept]
        if culled:
            facts.append(f"{culled} dropped for citing no evidence")
        return BlockResult({"attributes": kept},
                           f"{len(kept)} attributes, {len(strong)} strong", facts)


class GapFinder(Block):
    """
    Picks what to learn next. Rejects targets the subject already holds at
    strong level - without that rule it recommends the popular framework in a
    language they already know, which is not growth.
    """
    name = "gap_finder"

    DEFAULT = """This subject wants to grow. Their current attributes:
{attributes}

First identify their 2-3 MOST DISTINCTIVE attributes - rank by rarity, not by
strength. Then pick {n} targets that go DEEPER into those distinctive areas.

Hard rule: reject any target that is merely a popular framework or library for
something they already have at "strong" level. That is sideways, not growth.

Set depth to "deep" for internals/systems targets and "shallow" for
framework/tooling targets.

"terms" is fed to a code search engine that ANDs every word, so keep it to
ONE or TWO words naming the technology or subject - "postgres", "query planner",
"docker networking". Never a comma-separated list, never a sentence.

Reply with ONLY a JSON array:
[{{"target": "...", "why": "one sentence", "builds_on": "the attribute it deepens",
   "depth": "deep" or "shallow", "terms": "one or two words"}}]"""

    def run(self, state: dict) -> BlockResult:
        targets = parse_json(self.llm(self.prompt(
            self.DEFAULT,
            attributes=json.dumps(state["attributes"], indent=2),
            n=self.config.get("n", 3),
        )))
        strong = {a["name"].lower() for a in state["attributes"]
                  if a.get("level") == "strong"}
        kept, rejected = [], []
        for t in targets:
            t["terms"] = as_text(t.get("terms"))
            if t["target"].strip().lower() in strong:
                rejected.append(t["target"])
            else:
                kept.append(t)
        facts = [f"{t['target']} ({t.get('depth','?')}, builds on: {t.get('builds_on')})"
                 for t in kept]
        facts += [f"REJECTED {r}: already strong" for r in rejected]
        return BlockResult({"targets": kept}, f"{len(kept)} growth targets", facts)


class ParallelSearcher(Block):
    """One copy per target, concurrently. The search itself is the domain's job."""
    name = "parallel_searcher"
    parallel = True
    fan_out_over = "targets"

    def run(self, state: dict) -> BlockResult:
        target = state["_item"]
        try:
            items = self.domain.search(target)
        except Exception as e:
            return BlockResult({}, "", [f"{target['target']}: search failed - {e}"])
        for i in items:
            i["found_for"] = target["target"]
        items = items[: self.config.get("cap_per_target", 8)]
        return BlockResult({"candidates": items}, "",
                           [f"{target['target']}: {len(items)} candidates"])


class JunkFilter(Block):
    """
    Quality gate. Rules come from the caller, so the same block filters spam
    repos, dead CFPs, or anything else.

    Also caps items per group across the whole run, which stops one prolific
    source dominating the final list.
    """
    name = "junk_filter"

    def run(self, state: dict) -> BlockResult:
        rules = self.config.get("rules", [])
        cap = self.config.get("max_per_group")
        group_key = self.config.get("group_key")

        kept, dropped, counts, seen_titles = [], [], {}, {}

        # Identical titles across different groups are a clone-farm signal.
        for item in state.get("candidates", []):
            seen_titles[item.get("title", "")] = seen_titles.get(item.get("title", ""), 0) + 1

        for item in state.get("candidates", []):
            title = item.get("title", "")
            if seen_titles.get(title, 0) >= 3:
                dropped.append(f"{title[:44]}… (same title in 3+ sources)")
                continue

            signals = self.domain.quality_signals(item)
            reason = next((r for r in (rule(item, signals) for rule in rules) if r), None)
            if reason:
                dropped.append(f"{title[:44]}… ({reason})")
                continue

            if cap and group_key:
                g = item.get(group_key)
                counts[g] = counts.get(g, 0) + 1
                if counts[g] > cap:
                    dropped.append(f"{g} (cap of {cap} per source reached)")
                    continue

            item["signals"] = signals
            kept.append(item)

        return BlockResult({"vetted": kept},
                           f"{len(kept)} kept, {len(dropped)} dropped", dropped[:8])


class AuditedRanker(Block):
    """
    Scores candidates, then re-reads the attribute list and drops anything it
    cannot justify. The audit pass is the reason this is one block and not two.
    """
    name = "audited_ranker"

    SCORE = """Score each candidate for this subject.

SUBJECT ATTRIBUTES:
{attributes}

CANDIDATES:
{candidates}

For each candidate give:
- fit_score 0-100: could they be productive on this today?
- stretch_score 0-100: does it teach them something new and adjacent?

fit and stretch are INDEPENDENT axes. They must NOT sum to 100. A well-scoped
bug in a codebase they know well is high fit and low stretch. A substantial
feature in an unfamiliar subsystem of a familiar language is high on BOTH -
that is the ideal recommendation. Something requiring skills they lack is low
on both, because they cannot learn from work they cannot start. If every pair
you produce sums to 100 you are scoring one axis twice; redo it.

- relevance: does this candidate actually belong to the growth area named in
  its found_for field? Say "on-target" or "off-target" and be honest. Search
  broadens when a narrow query returns nothing, so some candidates drift.
- reasoning: MUST cite something specific from that candidate's own content -
  a function name, an error, a symptom, a concrete detail. Generic reasoning
  will be rejected by the auditor.
- gained: a concrete technique (e.g. "copy-on-write fork semantics",
  "predicate pushdown"). Never a product or repository name.

Reply with ONLY a JSON array:
[{{"id": "...", "title": "...", "group": "...", "fit_score": 0, "stretch_score": 0,
   "relevance": "on-target" or "off-target",
   "reasoning": "...", "attributes_used": ["..."], "gained": "..."}}]"""

    AUDIT = """You are auditing another agent's recommendations. Be strict.

THE SUBJECT'S ACTUAL ATTRIBUTES:
{attributes}

PROPOSED RECOMMENDATIONS:
{proposed}

Reject any recommendation where:
- attributes_used names something NOT in the attribute list above, or
- the reasoning is generic enough to apply to any subject, or
- two recommendations share the same reasoning template with only a name
  swapped, or
- "gained" is just a product, brand or repository name rather than a technique,
  or
- the reasoning itself admits the subject lacks what the candidate requires, or
- relevance is "off-target" - a candidate that drifted away from the growth
  area is not a recommendation, however interesting it looks, or
- fit_score and stretch_score sum to exactly 100, which means they were scored
  as one slider rather than two independent axes, or
- the title is too vague to act on (a single generic word, or under four
  words with no specific symptom, file or component named).

Reply with ONLY a JSON array of the survivors, unchanged except that you may
lower an inflated fit_score. Drop the rest entirely."""

    def run(self, state: dict) -> BlockResult:
        items = state.get("vetted", [])
        if not items:
            return BlockResult({"ranked": []}, "nothing to rank")

        attrs = json.dumps(state["attributes"], indent=2)
        payload = json.dumps([
            {"id": i.get("id"), "title": i.get("title"), "group": i.get("group"),
             "found_for": i.get("found_for"), "signals": i.get("signals"),
             "body": (i.get("body") or "")[:1000]}
            for i in items
        ], indent=2, default=str)[:14000]

        scored = parse_json(self.llm(self.prompt(
            self.SCORE, attributes=attrs, candidates=payload)))
        survivors = parse_json(self.llm(self.AUDIT.format(
            attributes=attrs, proposed=json.dumps(scored, indent=2))))

        survivors.sort(key=lambda m: m["fit_score"] + m["stretch_score"], reverse=True)
        top = survivors[: self.config.get("top_n", 5)]
        return BlockResult(
            {"ranked": top},
            f"{len(top)} recommendations ({len(scored) - len(survivors)} dropped on audit)",
            [f"{m['fit_score']}/{m['stretch_score']} — {m.get('title', m.get('id'))}"
             for m in top],
        )
