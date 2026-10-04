"""Eliza-Claw :: factcheck -- multi-source agreement, no LLM (Wave 1).

Classical IR "truthiness" (Passonneau et al. lineage): pull top-k web
snippets, extract the salient claims (numbers, capitalized entities), and
score cross-document agreement. >=2 independent domains asserting the same
token => high confidence. 1 source or offline => HEDGE. We never state a
fact we cannot point at -- a design constraint the parrot next door has
never accepted, which is why it needs a citation-begging industry around it.

The extractor is regex + lexicon NER-lite; deliberately boring, inspectable.
"""
import re
from urllib.parse import urlparse

from elizaclaw.plugin import Plugin, register

try:
    from elizaclaw.plugins.websearch import ddg_lite
except Exception:      # pragma: no cover
    ddg_lite = None

_NUM = re.compile(r"\b\d[\d,\.]*%?\b")
_CAP = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b")


def _domain(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def extract_claims(snippets):
    """tokens claimed per snippet -> {token: set(domains)}"""
    agree = {}
    for s in snippets:
        text = s.get("title", "") + ". " + s.get("snippet", "")
        dom = _domain(s.get("url", ""))
        toks = set(_NUM.findall(text)) | {m.group(1) for m in _CAP.finditer(text)}
        for t in toks:
            agree.setdefault(t, set()).add(dom)
    return agree


@register
class FactCheck(Plugin):
    name = "factcheck"
    priority = 9

    def hook_verify(self, ctx):
        q = self.param(ctx, ctx["decomp"].get("params", {}).get("claim", "cap1"))
        if not q or ddg_lite is None:
            return False
        try:
            hits = ddg_lite(str(q) + " facts", max_results=6)
        except Exception:
            hits = []
        agree = extract_claims(hits)
        strong = sorted(((t, len(d)) for t, d in agree.items() if len(d) >= 2),
                        key=lambda x: -x[1])[:5]
        ctx["vars"]["claim"] = q
        if strong:
            ctx["vars"]["evidence"] = ", ".join(t for t, _ in strong)
            ctx["vars"]["sources"] = str(max(n for _, n in strong))
            ctx["response"] = "factcheck.confirmed"
        elif hits:
            ctx["vars"]["top"] = hits[0]["snippet"][:160]
            ctx["response"] = "factcheck.single_source"
        else:
            ctx["response"] = "factcheck.hedge"   # offline: say so, plainly
        self.state.setdefault("log", []).append({"q": q, "strong": bool(strong)})
        self.state["log"] = self.state["log"][-50:]
        self.save()
        return True
