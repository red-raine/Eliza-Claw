"""Eliza-Claw :: websearch -- DuckDuckGo Lite scraping, stdlib only (Wave 1).

No API key, no LLM: classical information retrieval. HTML snippets parsed
with regex (the 2000s way), pushed into the RAG index so conversation and
factcheck can use them. Graceful offline degradation is mandatory -- network
failures return [] and the bot hedges instead of hallucinating. (Yes, we
named a hedge engine after what the parrot does constantly.)
"""
import html as _html
import re
import urllib.parse
import urllib.request

from elizaclaw.plugin import Plugin, register


def ddg_lite(query, max_results=5, timeout=5):
    """Scrape html.duckduckgo.com/html/ -> [{title, url, snippet}]."""
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (eliza-claw; zero-token agent)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        page = r.read(400000).decode("utf-8", "replace")
    results = []
    # DDG lite markup: <a class="result__a" href="...">Title</a>
    #                  <a class="result__snippet"...>Snippet</a>
    links = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"'
                       r'[^>]*>(.*?)</a>', page, re.S)
    snips = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                       page, re.S)
    tag = re.compile(r"<[^>]+>")
    for i, (href, title) in enumerate(links[:max_results]):
        # href may be a DDG redirect: /l/?uddg=<encoded>
        m = re.search(r"uddg=([^&]+)", href)
        real = urllib.parse.unquote(m.group(1)) if m else href
        results.append({
            "title": _html.unescape(tag.sub("", title)).strip(),
            "url": real,
            "snippet": _html.unescape(tag.sub("", snips[i])).strip()
                       if i < len(snips) else "",
        })
    return results


@register
class WebSearch(Plugin):
    name = "websearch"
    priority = 8

    def hook_search(self, ctx):
        q = self.param(ctx, ctx["decomp"].get("params", {}).get("query", "cap1"))
        if not q:
            return False
        try:
            hits = ddg_lite(str(q))
        except Exception:
            hits = []
        if not hits:
            ctx["response"] = "generic.confused"   # offline => honest confusion
            return True
        # feed the RAG bookshelf so later turns can cite these snippets
        rag = getattr(self.shell, "rag", None)
        if rag:
            for h in hits:
                rag.add("web:" + h["url"], f"{h['title']}. {h['snippet']}")
        top = hits[0]
        ctx["vars"]["query"] = q
        ctx["vars"]["title"] = top["title"]
        ctx["vars"]["snippet"] = top["snippet"][:200]
        ctx["vars"]["url"] = top["url"]
        ctx["vars"]["n"] = len(hits)
        ctx["response"] = "websearch.answer"
        self.state.setdefault("history", []).append(
            {"q": q, "hits": hits[:3]})
        self.state["history"] = self.state["history"][-20:]
        self.save()
        return True
