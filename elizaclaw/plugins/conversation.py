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
                    self.alias_map.setdefault(a.lower(), []).append(t)
        # lazy singleton tiered memory + rag shared across plugins
        self.mem = getattr(shell, "tiers", None)
        self.rag = getattr(shell, "rag", None)

    # ------------------------------------------------------------------ hooks
    def hook_chat(self, ctx):
        text = ctx["sentence"]
        topic = self.detect_topic(text, ctx)
        if not topic:
            # Wave 1 fallback: small talk still flows through memory tiers so
            # the STM/MTM/episodic record keeps growing even on chit-chat.
            self._ensure_backends()
            if self.mem:
                self.mem.add_turn("user", text)
                try:
                    val = shell_sentiment(self.shell, text)
                except Exception:
                    val = 0.0
                self.mem.record_episode(text, topic="(untopic)", valence=val)
            return False               # let ELIZA mirror-reflect instead
        self._ensure_backends()
        valence = 0.0
        try:
            valence = shell_sentiment(self.shell, text)
        except Exception:
            pass
        # --- write memory tiers
        if self.mem:
            self.mem.add_turn("user", text)
            self.mem.record_episode(text, topic=topic["name"], valence=valence)
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
        # exact alias hit wins, most aliases first
        best, best_n = None, 0
        for alias, tlist in self.alias_map.items():
            if alias in toks or (" " in alias and alias in text):
                for t in tlist:
                    n = sum(1 for a in t["aliases"] if a in toks)
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
