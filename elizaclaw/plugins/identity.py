"""Identity plugin: greet, recognize, and learn who is talking.

Uses the known-user graph seeded from example-users.json (aliases +
relationships). On 'hello' it greets by name if recognized; on "my name is X"
it registers/updates the profile. Facts learned during conversation persist
into memory/user.json -- that is the whole trick behind Eliza's illusion of
memory: you are just reading back what was stored.
"""
import re
from elizaclaw.plugin import Plugin, register

NAME_RX = re.compile(
    r"(?:my name is|i am called|call me|i'm|i am)\s+([a-z][\w'-]*)", re.I)
LOC_RX = re.compile(r"i live in\s+([a-z][\w' -]*?)(?:[.,!?]|$)", re.I)


@register
class Identity(Plugin):
    name = "identity"
    priority = 100

    def hook_identify(self, ctx):
        u = self.shell.state.user
        s = ctx["sentence"]
        # explicit self-report wins: "my name is alice"
        m = NAME_RX.search(ctx["sentence"])
        if m and not re.search(r"\byour name\b|\bthe name\b", s, re.I):
            name = m.group(1).capitalize()
            if name.lower() in ("not", "so", "really", "here", "fine",
                                "good", "back", "sorry", "feeling", "thinking"):
                pass                                  # "I'm fine" != a name
            else:
                return self.register(name, ctx)
        loc = LOC_RX.search(ctx["sentence"])
        if loc:
            u["location"] = loc.group(1).strip().capitalize()
            self.shell.state.save("user", u)
            ctx["vars"]["location"] = u["location"]
            ctx["response"] = "generic.remember_location"
            return True
        # asking about names?  'what is your name' -> the bot answers about
        # ITSELF (classic ELIZA had no self-model; Wave 0 gives it one).
        if re.search(r"\byour name\b", s, re.I):
            ctx["vars"]["name"] = self.shell.cfg["config"].get(
                "bot_name", "Eliza-Claw")
            ctx["response"] = "identity.self_name"
            return True
        if re.search(r"\b(my|your)\s+name\b|\bwho\s+are\s+(you|i)\b", s, re.I):
            if u.get("name"):
                ctx["vars"]["name"] = u["name"]
                ctx["response"] = "identity.greet_known"
            else:
                ctx["response"] = "identity.ask_name"
            return True
        if u.get("name"):
            ctx["vars"]["name"] = u["name"]
            ctx["response"] = "identity.greet_known"
            return True
        # try to recognize from aliases anywhere in the sentence
        for tok in ctx["tokens"]:
            known = self.shell.state.find_user(tok)
            if known:
                return self.adopt(known, ctx)
        ctx["response"] = "identity.greet_new"
        return True

    def hook_register(self, ctx):
        m = NAME_RX.search(ctx["sentence"])
        name = m.group(1).capitalize() if m else \
            (ctx["tokens"][-1] if ctx["tokens"] else None)
        if name:
            return self.register(name, ctx)
        return False

    # -------------------------------------------------------------- helpers
    def adopt(self, known, ctx):
        u = self.shell.state.user
        u.update({"name": known["name"],
                  "location": known.get("location"),
                  "interests": known.get("interests", []),
                  "id": known.get("id")})
        u.setdefault("history", {}).update(known.get("history", {}))
        u["facts"] = {f: True for f in known.get("facts_known", [])}
        self.shell.state.save("user", u)
        cur = self.shell.state.seed_users()
        cur["current"] = known["id"]
        self.shell.state.save("users", cur)
        ctx["vars"]["name"] = known["name"]
        ctx["response"] = "identity.greet_known"
        return True

    def register(self, name, ctx):
        u = self.shell.state.user
        known = self.shell.state.find_user(name)
        if known and known["name"].lower() == name.lower():
            return self.adopt(known, ctx)
        fresh = not u.get("name")
        u["name"] = name
        self.shell.state.save("user", u)
        ctx["vars"]["name"] = name
        ctx["response"] = "identity.name_learned" if fresh \
            else "identity.greet_known"
        self.shell.state.log("learn", "name=%s" % name)
        return True
