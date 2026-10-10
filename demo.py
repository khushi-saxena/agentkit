"""
Offline demo: runs BOTH products through the identical six blocks.

    python demo.py

No API keys, no network - it uses the test stubs. The point is to show that
the same chain produces two different products with only the adapter swapped.
"""
from agentkit import Pipeline
from agentkit.blocks import (AuditedRanker, EvidenceAssessor, GapFinder,
                             JunkFilter, ParallelSearcher, ProfileBuilder)
from adapters import github_oss
from adapters.conference_cfp import RULES as CFP_RULES
from adapters.conference_cfp import CFPDomain
from tests.stubs import CFPS, StubGitHub, speaker_source, stub_llm


def chain(rules):
    return [ProfileBuilder(), EvidenceAssessor(), GapFinder(n=2),
            ParallelSearcher(cap_per_target=8),
            JunkFilter(rules=rules, group_key="group", max_per_group=2),
            AuditedRanker(top_n=5)]


def show(title, pipeline):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    for name, result in pipeline.stream({"subject_id": "khushi-saxena",
                                         "extra_context": "UBS, DRDO, LogLens"}):
        print(f"  {name:20s} {result.summary}")
        for fact in result.facts[:4]:
            print(f"      - {fact}")


show("PRODUCT 1: open-source contribution matcher",
     Pipeline(blocks=chain(github_oss.RULES), domain=StubGitHub(), llm=stub_llm))

show("PRODUCT 2: conference CFP matcher   (IDENTICAL BLOCKS)",
     Pipeline(blocks=chain(CFP_RULES),
              domain=CFPDomain(cfp_source=lambda: CFPS,
                               speaker_source=speaker_source),
              llm=stub_llm))

print("\nSame six blocks both times. Only the adapter differed.")
