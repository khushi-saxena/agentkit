"""
Pinpoint where the discovery funnel empties, without burning a full LLM run.

    python diagnose.py "database internals"
    python diagnose.py "SwiftUI" --shallow

Hits GitHub only - no model calls, so it costs nothing and runs in seconds.
Prints every stage: repos matched, which passed the health check and why the
rest failed, and how many open issues each healthy repo actually has.
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from adapters.github_oss import GitHubOSSDomain   # noqa: E402


def main(terms: str, deep: bool):
    domain = GitHubOSSDomain()
    print(f"terms      : {terms!r}")
    print(f"path       : {'deep (no label filter)' if deep else 'shallow (good first issue)'}")
    print(f"min_stars  : {domain.min_stars}")
    print(f"health     : rejects <{domain.min_stars // 5} stars or <3 of last 20 PRs merged")
    print("-" * 68)

    repos = domain.find_repos(terms)
    print(f"{len(repos)} repos matched\n")
    if not repos:
        print("Nothing matched. Try broader terms, or lower min_stars:")
        print("  GitHubOSSDomain(min_stars=200)  in app/pipeline.py")
        return

    total = 0
    for repo in repos[: domain.health_checks]:
        health = domain.check_repo_health(repo["full_name"])
        if not health["ok"]:
            print(f"  REJECTED  {repo['full_name']:<40} {health['reason']}")
            continue
        issues = domain.list_repo_issues(repo["full_name"], deep)
        total += len(issues)
        print(f"  HEALTHY   {repo['full_name']:<40} "
              f"{health['stars']:>6} stars, {health['merged_recently']}/20 merged, "
              f"{len(issues)} issues")
        for issue in issues[:3]:
            print(f"              - {issue['title'][:62]}")

    print("-" * 68)
    print(f"{total} candidate issues would reach the gatekeeper")
    if not total:
        print("\nRepos were fine but produced no issues. Most likely:")
        print("  - shallow path and nothing carries a 'good first issue' label")
        print("  - every open issue already has an assignee")
        print("Try the same terms with the deep path (no --shallow flag).")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "database internals",
         "--shallow" not in sys.argv)
