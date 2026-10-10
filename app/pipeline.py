"""
The product: an open-source contribution matcher, assembled from agentkit
blocks plus the GitHub adapter.

This is the whole configuration. Swap the domain and rules and the same chain
becomes a different product - see adapters/conference_cfp.py.
"""
from agentkit.blocks import (AuditedRanker, EvidenceAssessor, GapFinder,
                             JunkFilter, ParallelSearcher, ProfileBuilder)
from adapters.github_oss import RULES, GitHubOSSDomain

LABELS = {
    "profile_builder":   ("Profiler", "Reading your real GitHub repos"),
    "evidence_assessor": ("Skill Assessor", "Working out your skills, with evidence"),
    "gap_finder":        ("Growth Planner", "Choosing what you should learn next"),
    "parallel_searcher": ("Issue Scouts", "Searching GitHub live for open issues"),
    "junk_filter":       ("Gatekeeper", "Throwing out bounty farms and template junk"),
    "audited_ranker":    ("Matcher", "Scoring fit vs stretch, then auditing itself"),
}
ORDER = list(LABELS)


def build_blocks():
    return [
        ProfileBuilder(),
        EvidenceAssessor(count="6-10"),
        GapFinder(n=3),
        ParallelSearcher(cap_per_target=8),
        JunkFilter(rules=RULES, group_key="group", max_per_group=2),
        AuditedRanker(top_n=5),
    ]


def build_domain():
    return GitHubOSSDomain()
