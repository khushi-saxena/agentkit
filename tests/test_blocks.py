"""
Guard-rail tests: every rule that stops the system recommending nonsense.

Each one feeds the pipeline deliberately bad input and asserts it gets
rejected. These are the tests that matter - a happy-path test only proves the
plumbing runs.
"""
from agentkit import Pipeline
from agentkit.blocks import (AuditedRanker, EvidenceAssessor, GapFinder,
                             JunkFilter, ParallelSearcher, ProfileBuilder,
                             as_text, parse_json)
from adapters import github_oss
from adapters.conference_cfp import RULES as CFP_RULES
from adapters.conference_cfp import CFPDomain
from tests.stubs import (CFPS, StubGitHub, fenced_llm, speaker_source,
                         stub_llm)


def blocks(rules):
    return [ProfileBuilder(), EvidenceAssessor(), GapFinder(n=2),
            ParallelSearcher(cap_per_target=8),
            JunkFilter(rules=rules, group_key="group", max_per_group=2),
            AuditedRanker(top_n=5)]


def run_oss(llm=stub_llm):
    return Pipeline(blocks=github_oss.RULES and blocks(github_oss.RULES),
                    domain=StubGitHub(), llm=llm).run(
        {"subject_id": "khushi-saxena", "extra_context": "UBS, DRDO"})


# --- the guard rails -------------------------------------------------------

def test_attribute_without_evidence_is_culled():
    final = run_oss()
    names = [a["name"] for a in final["attributes"]]
    assert "Ghost Skill" not in names
    assert all(a["evidence"].strip() for a in final["attributes"])


def test_gap_finder_rejects_a_target_already_held_strongly():
    final = run_oss()
    targets = [t["target"] for t in final["targets"]]
    assert "Python" not in targets
    assert "database internals" in targets


def test_junk_filter_drops_bounty_farm_titles():
    final = run_oss()
    assert not any("ONBOARD" in i["title"] for i in final["vetted"])


def test_junk_filter_drops_thin_bodies():
    final = run_oss()
    assert not any(i["title"] == "Tiny body issue" for i in final["vetted"])


def test_junk_filter_caps_items_per_source():
    """One prolific repo must not supply the entire recommendation list."""
    final = run_oss()
    from collections import Counter
    counts = Counter(i["group"] for i in final["vetted"])
    assert all(c <= 2 for c in counts.values()), counts


def test_audit_drops_recommendation_citing_an_absent_skill():
    final = run_oss()
    allowed = {a["name"] for a in final["attributes"]}
    assert final["ranked"], "pipeline produced nothing at all"
    for rec in final["ranked"]:
        assert set(rec["attributes_used"]) <= allowed, rec


def test_identical_titles_across_sources_are_dropped():
    """Clone farms publish the same issue text under many repo names."""
    class CloneFarm(StubGitHub):
        def search(self, target):
            return [{"id": f"https://gh/{n}", "group": f"clone{n}/repo",
                     "title": "Enatega Customer Application: review a past order",
                     "body": "x" * 400, "comments": 2, "repo_stars": 900,
                     "updated_at": "2026-10-06T00:00:00Z"} for n in range(4)]

    final = Pipeline(blocks=blocks(github_oss.RULES),
                     domain=CloneFarm(), llm=stub_llm).run(
        {"subject_id": "x", "extra_context": ""})
    assert final["vetted"] == []


# --- parsing robustness ----------------------------------------------------

def test_fenced_json_is_parsed():
    final = run_oss(llm=fenced_llm)
    assert final["ranked"]


def test_keyword_list_is_coerced_to_text():
    """GPT-4o returns search terms as an array; the searcher needs a string."""
    assert as_text(["storage", "engine"]) == "storage engine"
    assert as_text("storage engine") == "storage engine"
    assert as_text(None) == ""


def test_parse_json_handles_bare_and_fenced():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


# --- the reuse claim -------------------------------------------------------

def test_same_blocks_power_a_completely_different_product():
    domain = CFPDomain(cfp_source=lambda: CFPS, speaker_source=speaker_source)
    final = Pipeline(blocks=blocks(CFP_RULES), domain=domain, llm=stub_llm).run(
        {"subject_id": "khushi", "extra_context": "UBS, DRDO"})

    kept = [i["group"] for i in final["vetted"]]
    assert kept == ["PyCon US"], kept          # SREcon too soon, DeadConf passed
    assert final["ranked"], "second product produced nothing"


def test_both_products_enforce_the_same_guard_rails():
    """The rules live in the blocks, so they apply to every domain."""
    domain = CFPDomain(cfp_source=lambda: CFPS, speaker_source=speaker_source)
    cfp = Pipeline(blocks=blocks(CFP_RULES), domain=domain, llm=stub_llm).run(
        {"subject_id": "x", "extra_context": ""})
    assert "Ghost Skill" not in [a["name"] for a in cfp["attributes"]]
    assert "Python" not in [t["target"] for t in cfp["targets"]]


# --- scoring quality -------------------------------------------------------

def test_audit_rejects_scores_that_sum_to_one_hundred():
    """
    A real run produced 90/10, 85/15, 70/30, 60/40, 75/25 - every pair summing
    to exactly 100, which means fit and stretch were scored as one slider.
    """
    import json
    from agentkit.blocks import AuditedRanker

    recs = [{"id": "u1", "title": "Replica crash after fork failure",
             "group": "redis/redis", "fit_score": 90, "stretch_score": 10,
             "relevance": "on-target", "reasoning": "Cites the SIGABRT path.",
             "attributes_used": ["Python"], "gained": "fork semantics"}]

    captured = {}

    def llm(prompt):
        if "Score each candidate" in prompt:
            return json.dumps(recs)
        captured["audit"] = prompt
        return json.dumps([])          # auditor rejects it

    block = AuditedRanker()
    block.domain = type("D", (), {"prompts": lambda s: {}})()
    block.llm = llm
    out = block.run({"vetted": [{"id": "u1", "title": "t", "group": "g",
                                 "body": "b", "found_for": "x"}],
                     "attributes": [{"name": "Python", "level": "strong",
                                     "evidence": "e"}]})

    assert "sum to exactly 100" in captured["audit"]
    assert out.updates["ranked"] == []


def test_scoring_prompt_states_the_axes_are_independent():
    from agentkit.blocks import AuditedRanker
    assert "INDEPENDENT axes" in AuditedRanker.SCORE
    assert "must NOT sum to 100" in AuditedRanker.SCORE


def test_audit_rejects_off_target_candidates():
    from agentkit.blocks import AuditedRanker
    assert "off-target" in AuditedRanker.AUDIT


def test_vague_titles_are_filtered():
    """A run recommended an issue titled 'farm'."""
    from adapters.github_oss import substantive_title
    assert substantive_title({"title": "farm"}, {})
    assert substantive_title({"title": "jcenter() error"}, {})
    assert substantive_title(
        {"title": "KSampler fails with CUDA HIPBLAS error"}, {}) is None
