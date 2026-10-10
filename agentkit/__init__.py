"""
agentkit - reusable blocks for multi-agent pipelines where agents check each
other's work.

    from agentkit import Pipeline, make_llm
    from agentkit.blocks import ProfileBuilder, EvidenceAssessor, AuditedRanker
    from adapters.github_oss import GitHubOSSDomain, RULES

    pipeline = Pipeline(blocks=[...], domain=GitHubOSSDomain(), llm=make_llm())
    for name, result in pipeline.stream({"subject_id": "khushi-saxena"}):
        print(name, result.summary)

Same blocks run on LangGraph instead:

    from agentkit.executors.langgraph_runner import build_graph, initial_state
    graph = build_graph([...], domain, llm)
    final = graph.invoke(initial_state("khushi-saxena"))
"""
from .core import Block, BlockResult, Domain, Pipeline, merge
from .llm import make_llm

__all__ = ["Block", "BlockResult", "Domain", "Pipeline", "merge", "make_llm"]
__version__ = "0.1.0"
