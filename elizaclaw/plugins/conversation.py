"""Eliza-Claw :: conversation -- Wave 1's deep-chat plugin.

The heart of "1-2k common human topics with follow-up replies". Pipeline per
turn that reaches xnone (nothing more specific claimed it):

  1. topic detect: alias lookup over data/topics.json (+ parent sub-topics),
     boosted by WordNet synonyms via shell.lexicon, fallback = RAG best hit
  2. memory tiers: every turn -> STM/MTM; sentiment valence -> episodic tag
  3. RAG: retrieve a passage (topic doc / episode / fact) and weave it into
     the reply template -- retrieval-augmented, zero tokens
  4. follow-up ladder: walk the topic's Aron-style chain light->deep; if the
     user stalls two turns in, fire a generic deepener instead

If the corpus can't help, we fall through to classic ELIZA reflection so the
bot never goes silent.
"""
import json
import os
import random

from elizaclaw.plugin import Plugin, register

HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS_FILE = os.path.join(HERE, "..", "..", "data", "topics.json")


@register
def _lemma_variants(word):
    """Tiny morphology folds (stdlib stand-in for a lemmatizer)."""
    out = {word}
    if word.endswith("ies") and len(word) > 4:
        out.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 4:
        out.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        out.add(word[:-1])
    if not word.endswith("s"):
        out.add(word + "s")
    return out


class Conversation(Plugin):
    name = "conversation"
    priority = -5                      # last resort before raw ELIZA

    def __init__(self, shell):
        super().__init__(shell)
        self.topics = []
        self.alias_map = {}            # alias -> list of topic dicts
        if os.path.exists(TOPICS_FILE):
            with open(TOPICS_FILE) as f:
                self.topics = json.load(f)["topics"]
            for t in self.topics:
                for a in t["aliases"] + [t["name"]]:
                    a = a.lower()
                    self.alias_map.setdefault(a, []).append(t)
                    # morphology fold: 'dogs' also indexes the 'dog' alias
                    for v in _lemma_variants(a):
                        if v != a:
                            self.alias_map.setdefault(v, []).append(t)
        # lazy singleton tiered memory + rag shared across plugins
        self.mem = getattr(shell, "tiers", None)
        self.rag = getattr(shell, "rag", None)

    # ------------------------------------------------------------- lifecycle
    def on_turn(self):
        """Wave 1 (bug fix): record EVERY user turn into STM/MTM.

        Previously add_turn only fired when this plugin claimed the turn,
        so command turns ('remind me to...', 'search for...') never entered
        the mid-term store and RAG could not retrieve them later. The shell
        calls on_turn after each user input regardless of routing.
        """
        if self.mem is None:
            self._ensure_backends()
        last_user = None
        for item in reversed(getattr(self.mem, "stm", []) if self.mem else []):
            if item.get("role") == "user":
                last_user = item["text"]
                break
        hist = self.shell.state.log_tail(2) if hasattr(
            self.shell.state, "log_tail") else []
        for role, text in hist:
            if role == "user" and text != last_user and self.mem:
                self.mem.add_turn("user", text)
                break

    # ------------------------------------------------------------------ hooks
    def hook_chat(self, ctx):
        text = ctx["sentence"]
        topic = self.detect_topic(text, ctx)
        self._ensure_backends()
        valence = 0.0
        try:
            valence = shell_sentiment(self.shell, text)
        except Exception:
            pass
        if not topic:
            # Wave 1 fallback: small talk still flows through memory tiers so
            # the STM/MTM/episodic record keeps growing even on chit-chat.
            # Bug fix (wave 1 e2e): only write if on_turn hasn't already
            # recorded this exact turn -- prevents duplicate episodes.
            if self.mem:
                last_user = None
                for item in reversed(self.mem.stm):
                    if item.get("role") == "user":
                        last_user = item["text"]
                        break
                if last_user != text:
                    self.mem.add_turn("user", text)
                dup = any(e["what"] == text and e.get("topic") == "(untopic)"
                          for e in self.mem.epi[-8:])
                if not dup:
                    self.mem.record_episode(text, topic="(untopic)",
                                            valence=valence)
            return False               # let ELIZA mirror-reflect instead
        # --- write memory tiers (dedup vs on_turn, same guard as above)
        if self.mem:
            last_user = None
            for item in reversed(self.mem.stm):
                if item.get("role") == "user":
                    last_user = item["text"]
                    break
            if last_user != text:
                self.mem.add_turn("user", text)
            dup = any(e["what"] == text and e.get("topic") == topic["name"]
                      for e in self.mem.epi[-8:])
            if not dup:
                self.mem.record_episode(text, topic=topic["name"],
                                        valence=valence)
        # --- pick the next rung on the follow-up ladder
        prog = self.state.setdefault("ladder", {})
        step = prog.get(topic["name"], 0)
        chain = topic["follow_ups"] or topic["openers"]
        question = chain[min(step, len(chain) - 1)]
        prog[topic["name"]] = step + 1
        # stall detection: short answers => escalate with a generic deepener
        if len(ctx["tokens"]) <= 3 and step >= 2 and topic.get("deepeners"):
            d = topic["deepeners"]
            used = self.state.setdefault("deep_used", [])
            pool = [x for i, x in enumerate(d) if i not in used] or d
            question = random.choice(pool)
            self.state["deep_used"] = used + [d.index(question)]
        self.save()
        # --- RAG weave: retrieved passage becomes an aside in the reply
        aside = ""
        if self.rag:
            try:
                hits = self.rag.search(text, k=2)
                mem_hits = [d for s, d in hits
                            if d["meta"].get("kind") in ("mtm", "episode", "fact")]
                if mem_hits:
                    aside = f"(Earlier you mentioned: {mem_hits[0]['text'][:80]}.) "
            except Exception:
                pass
        ctx["vars"]["aside"] = aside
        ctx["vars"]["question"] = question
        ctx["vars"]["topic"] = topic["name"]
        ctx["response"] = "conversation.deeptopic"
        if self.mem:
            self.mem.add_turn("bot", question)
        return True

    # ------------------------------------------------------------- topic detect
    def detect_topic(self, text, ctx):
        toks = set(ctx["tokens"])
        folded = {v for a in toks for v in _lemma_variants(a)}
        # exact alias hit wins; multi-word aliases match as substrings so
        # sub-topics ('movie night') beat their parents on the full phrase
        best, best_n = None, 0
        for alias, tlist in self.alias_map.items():
            single = " " not in alias
            hit = (alias in folded if single else alias in text)
            if not hit:
                continue
            for t in tlist:
                # count how many of this topic's aliases the sentence hits;
                # single words match folded tokens, phrases match substring.
                # (bug fix e2e: the old one-liner `if x if y else z` inside a
                # generator was invalid Python -- syntax error at import.)
                n = sum(1 for a in t["aliases"]
                        if ((a in folded) if " " not in a else (a in text)))
                if len(alias.split()) > 1:
                    n += 2                       # phrase alias bonus
                if n > best_n:
                    best, best_n = t, n
        if best:
            return best
        # WordNet-synonym growth: expand sentence seeds, re-probe aliases
        lex = getattr(self.shell, "lexicon", None)
        if lex is not None:
            try:
                grown = lex.expand_synset(list(toks))
                for alias, tlist in self.alias_map.items():
                    if alias in grown:
                        return tlist[0]
            except Exception:
                pass
        # RAG fallback: nearest indexed topic doc
        if self.rag and self.rag.docs:
            try:
                d = self.rag.best(text)
                if d and d["meta"].get("kind") == "topic":
                    return d["meta"]["topic"]
            except Exception:
                pass
        return None

    def _ensure_backends(self):
        if self.mem is None:
            self.mem = getattr(self.shell, "tiers", None)
        if self.rag is None:
            self.rag = getattr(self.shell, "rag", None)


def shell_sentiment(shell, text):
    fn = getattr(shell.lexicon, "sentiment", None)
    return fn(text) if fn else 0.0
