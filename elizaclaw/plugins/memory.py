"""Eliza-Claw :: memory plugin -- commands over the four tiers (Wave 1).

'remember that X' -> LTM fact, 'what do you remember' -> dump,
'memory stats' -> tier sizes. Also the heartbeat owner of decay + RAG sync:
every tick we ACT-R-decay MTM/episodic and re-feed the retrieval index, so
the bookshelf stays current without any model in the loop.
"""
import re

from elizaclaw.plugin import Plugin, register


@register
class MemoryTiersPlugin(Plugin):
    name = "memory_tiers_plugin"
    priority = 8

    def __init__(self, shell):
        super().__init__(shell)
        # one shared instance for the whole process; shell exposes it too
        from elizaclaw.memory_tiers import MemoryTiers
        if not hasattr(shell, "tiers"):
            shell.tiers = MemoryTiers(shell.state)
        self.mem = shell.tiers
        from elizaclaw.rag import Rag
        if not hasattr(shell, "rag"):
            shell.rag = Rag()
            try:
                shell.rag.load_corpus()
            except FileNotFoundError:
                pass                      # corpus not built yet -- fine
        self.rag = shell.rag

    # ------------------------------------------------------------------ hooks
    def hook_store(self, ctx):
        spec = ctx["decomp"].get("params", {}).get("text", "cap1")
        text = self.param(ctx, spec)
        # NLP slot refinement (Wave 1): if the raw capture is empty or a bare
        # preposition ('remind to call mom' -> cap1=''), grab the noun phrase
        # after the cue word instead -- POS-tagged, prep-bounded.
        toks = ctx.get("tokens") or []
        cue = {"remember", "that", "note"}
        weak = (not text or str(text).strip() == ""
                or set(str(text).lower().split()) <= cue)
        if weak:
            lex = getattr(self.shell, "lexicon", None)
            if lex is not None:
                try:
                    idx = max((i for i, t in enumerate(toks) if t in cue),
                              default=None)
                    if idx is not None:
                        np_ = lex.noun_phrase(toks, idx + 1)
                        if np_:
                            text = np_
                except Exception:
                    pass
            if weak:
                tail = " ".join(t for t in toks if t not in cue).strip()
                text = tail or None
        if not text:
            return False
        key = re.sub(r"[^a-z0-9]+", "_", str(text).lower())[:40] or f"fact{ctx['turn']}"
        self.mem.remember_fact(key, str(text))
        self.rag.add(f"ltm:{key}", f"Fact: {text}")
        ctx["vars"].update({"key": key, "value": text})
        ctx["response"] = "memory.remembered"
        return True

    def hook_show(self, ctx):
        facts = self.mem.facts()
        eps = self.mem.episodes(k=3)
        lines = [f"- {v}" for v in list(facts.values())[-8:]]
        lines += [f"- (episode, {e.get('topic','?')}) {e['what'][:60]}"
                  for e in eps]
        # 'what do you remember about X' -> RAG retrieval over every tier
        q = self.param(ctx, ctx["decomp"].get("params", {}).get("query"))
        if q:
            hits = [d for s, d in self.rag.search(str(q), k=4)
                    if d["meta"].get("kind") in ("fact", "episode", "mtm")]
            ctx["vars"]["query"] = q
            ctx["vars"]["facts"] = ("\n".join("- " + d["text"][:90]
                                              for d in hits)
                                    or "(nothing yet)")
            ctx["response"] = "memory.recall"
            return True
        ctx["vars"]["facts"] = "\n".join(lines) or "(nothing yet)"
        st = self.mem.stats()
        ctx["vars"].update({k: v for k, v in
                            zip(("stm", "mtm", "ltm", "epi"), st.values())})
        ctx["response"] = "memory.dump"
        return True

    def hook_stats(self, ctx):
        st = self.mem.stats()
        ctx["vars"].update({k: v for k, v in
                            zip(("stm", "mtm", "ltm", "epi"), st.values())})
        ctx["response"] = "memory.stats"
        return True

    # ------------------------------------------------------------- heartbeat
    def tick(self):
        """Decay activations + keep the RAG index synced with memory."""
        self.mem.decay(minutes=1.0)
        try:
            self.rag.sync_memory(self.mem)
        except Exception:
            pass
