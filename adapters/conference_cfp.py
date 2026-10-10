"""
Adapter: conference talk / CFP matcher.

A different product entirely - it profiles a speaker and finds calls-for-papers
they could credibly submit to. Same six blocks, same evidence enforcement, same
parallel fan-out, same self-audit.

This file is ~80 lines. That is the argument for the SDK: everything expensive
was already written and is shared.
"""
from __future__ import annotations


class CFPDomain:
    """
    Takes its CFP list and speaker history from callables, so you can swap in a
    live source (Sessionize, PaperCall, an RSS feed) without touching a block.
    """

    name = "conference-cfp"

    def __init__(self, cfp_source, speaker_source):
        self._cfps = cfp_source                # () -> list[dict]
        self._speaker = speaker_source         # (speaker_id) -> dict

    def fetch_subject(self, subject_id: str) -> dict:
        return self._speaker(subject_id)

    def search(self, target: dict) -> list[dict]:
        terms = (target.get("terms") or target.get("target", "")).lower()
        words = [w for w in terms.split() if len(w) > 3]
        out = []
        for cfp in self._cfps():
            haystack = f"{cfp['name']} {cfp.get('track','')} {cfp.get('topics','')}".lower()
            if words and not any(w in haystack for w in words):
                continue
            out.append({
                "id": cfp["url"], "group": cfp["name"],
                "title": f"{cfp['name']} — {cfp.get('track','General')}",
                "body": cfp.get("description", ""),
                "attendees": cfp.get("attendees", 0),
                "days_to_deadline": cfp.get("days_to_deadline", 0),
                "accepts_first_time": cfp.get("accepts_first_time", False),
            })
        return out

    def quality_signals(self, item: dict) -> dict:
        return {"attendees": item.get("attendees", 0),
                "days_left": item.get("days_to_deadline", 0),
                "body_len": len(item.get("body", "")),
                "first_time_friendly": item.get("accepts_first_time", False)}

    def prompts(self) -> dict[str, str]:
        # Domain wording where the generic default is too vague to be useful.
        return {
            "gap_finder": """This speaker wants to grow their speaking profile.
Their current strengths:
{attributes}

Pick {n} topic areas where they could give a talk that is credible today but
stretches them - adjacent to their deepest expertise, not a repeat of something
they could already deliver in their sleep.

Set depth to "deep" for internals/systems topics, "shallow" for intro topics.

Reply with ONLY a JSON array:
[{{"target": "...", "why": "one sentence", "builds_on": "the strength it draws on",
   "depth": "deep" or "shallow", "terms": "topic keywords"}}]"""
        }


def deadline_not_passed(item, signals):
    return "deadline passed" if signals["days_left"] <= 0 else None


def enough_time_to_write(item, signals):
    return ("under 10 days to write a talk"
            if 0 < signals["days_left"] < 10 else None)


def real_cfp(item, signals):
    return "description too thin" if signals["body_len"] < 80 else None


RULES = [deadline_not_passed, enough_time_to_write, real_cfp]
