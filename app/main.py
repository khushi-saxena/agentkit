"""
Run the matcher from the terminal on LangGraph.

    python -m app.main khushi-saxena

Put your resume in resume.txt and it gets folded into the profile - public
repos alone systematically underrate anyone whose real work is in private
company repos.
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agentkit import make_llm                                   # noqa: E402
from agentkit.executors.langgraph_runner import (build_graph,   # noqa: E402
                                                 initial_state)
from app.pipeline import LABELS, build_blocks, build_domain     # noqa: E402
from app.resume import load_resume                              # noqa: E402


def main(username: str):
    resume, source = load_resume()
    graph = build_graph(build_blocks(), build_domain(), make_llm())

    print("=" * 72)
    print(f"OPEN-SOURCE CONTRIBUTION MATCHER  -  github.com/{username}")
    print("=" * 72)
    if source:
        print(f"Resume: {os.path.basename(source)} ({len(resume):,} chars)\n")
    else:
        print("No resume found - assessing from public repos only, so expect\n"
              "generic growth targets. Drop a resume.pdf or resume.txt here.\n")

    final = graph.invoke(initial_state(username, resume))

    for step in final.get("trace", []):
        label = LABELS.get(step["block"], (step["block"], ""))[0]
        print(f"\n[{label}] {step['summary']}")
        for fact in step["facts"][:8]:
            print(f"    {fact}")

    print("\n" + "=" * 72)
    print("TOP MATCHES")
    print("=" * 72)
    matches = final.get("ranked", [])
    if not matches:
        print("\nNothing survived vetting. That is the system being strict, not\n"
              "broken - loosen the thresholds in GitHubOSSDomain to see more.")
        return
    for n, m in enumerate(matches, 1):
        print(f"\n{n}. {m.get('title')}")
        print(f"   {m.get('group')}  |  fit {m['fit_score']}  stretch {m['stretch_score']}")
        print(f"   {m.get('id')}")
        print(f"   uses: {', '.join(m.get('attributes_used', []))}")
        print(f"   learns: {m.get('gained')}")
        print(f"   why: {m.get('reasoning')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "khushi-saxena")
