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
        text = self.param(ctx, ctx["decomp"].get("params", {}).get("text", "cap1"))
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
        ctx["vars"]["facts"] = "\n".join(lines) or "(nothing yet)"
        ctx["response"] = None
        ctx["explicit_response"] = None
        # render inline: simplest template is generic.continue + our text
        ctx["vars"]["memory_dump"] = ctx["vars"]["facts"]
        ctx["response"] = "memory.stats"
        st = self.mem.stats()
        ctx["vars"].update({k: v for k, v in
                            zip(("stm", "mtm", "ltm", "epi"), st.values())})
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
