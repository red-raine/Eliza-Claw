"""Eliza conversation plugin: the 1966 soul of the machine.

Pattern-match -> extract a segment -> post-transform persons -> reflect back.
Plus the 'modern NLP' upgrade from pos_transformers: verb/object focus using
our POS-lite extractor instead of an NLTK dependency parse (same result,
zero downloads, zero tokens).
"""
from elizaclaw import nlu
from elizaclaw.plugin import Plugin, register


@register
class ElizaConversation(Plugin):
    name = "eliza_conversation"
    priority = 1

    def hook_reflect(self, ctx):
        cap = ctx["captures"][0] if ctx["captures"] else \
            " ".join(ctx["tokens"])
        flipped = nlu.reflect(cap, self.shell.cfg["transformers"]["post"])
        ctx["vars"].update({"segment": flipped, "topic": flipped.split()[0]
                            if flipped else "that"})
        # classic Eliza memory trick: stash notable segments for later recall
        mem = self.state.setdefault("reflections", [])
        mem.append(flipped)
        del mem[:-15]
        self.save()

    def hook_pos_reflect(self, ctx):
        """Verb-focus reflection: 'I want to finish the project' ->
        'What is stopping you from finishing it?'"""
        verb, obj = nlu.extract_verb_object(ctx["sentence"])
        if not verb:
            return False
        ctx["vars"]["verb"] = verb
        ctx["vars"]["object"] = obj or "it"
        ctx["response"] = "pos.verb_focus" if not obj else "pos.object_focus"
        tmpl_key = "reflection_templates.verb_focus"
        pool = self.shell.cfg.get("pos_transformers", {}).get(
            "reflection_templates", {})
        options = pool.get("verb_focus", [])
        if options:
            ctx["forced_template_list"] = options
        return True

    def learn_fact(self, ctx, category):
        """The 'learn' directive on decomps: store what the user reveals."""
        idx = None
        u = self.shell.state.user
        facts = u.setdefault("facts", {})
        seg = ctx["captures"][0] if ctx["captures"] else \
            " ".join(ctx["tokens"])
        if seg:
            facts[seg] = {"category": category, "turn": ctx["turn"]}
            self.shell.state.save("user", u)
            self.shell.state.log("learn", "%s: %s" % (category, seg))
        if category == "emotions":
            board = self.shell.state.board
            low = seg.lower()
            sad = set(self.shell.cfg["synonyms"].get("sad", []))
            happy = set(self.shell.cfg["synonyms"].get("happy", []))
            if any(w in sad for w in low.split()):
                board["inferred_state"]["emotional_state"] = "frustrated"
                ctx["vars"]["feeling"] = next((w for w in low.split()
                                               if w in sad), "down")
            elif any(w in happy for w in low.split()):
                board["inferred_state"]["emotional_state"] = "happy"
                ctx["vars"]["feeling"] = next((w for w in low.split()
                                               if w in happy), "good")
            self.shell.state.save("blackboard", board)
