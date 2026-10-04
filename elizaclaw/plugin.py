"""Eliza-Claw :: plugin -- the base class every skill inherits from.

A Plugin is four things, exactly as promised in the README:
  * keywords   -- patterns it matches        (from eliza.json "keys")
  * hooks      -- functions it can run       (methods named on_<hookname>)
  * templates  -- mad-lib responses          (eliza.json "responses" trees)
  * state      -- persistent JSON             (memory/plugins/<name>.json)

Registration is automatic: any Plugin subclass defined in a module of
elizaclaw/plugins/ registers itself with the shell when imported.
No framework, no entrypoints, no magic -- one registry list.
"""

REGISTRY = []


class Plugin:
    name = "plugin"
    priority = 0

    def __init__(self, shell):
        self.shell = shell
        self.state = shell.state.plugin_state(self.name)

    # ------------------------------------------------------------ lifecycle
    def save(self):
        self.shell.state.save_plugin(self.name, self.state)

    def tick(self):
        """Called by the heartbeat. Override if you need to wake up often."""

    def on_turn(self):
        """Called after each user turn. Override for metacognition."""

    # --------------------------------------------------------------- routing
    def handle(self, key, decomp, ctx):
        """Default dispatch: fire the decomp's hook if we implement it.

        Returns True if something actionable happened.
        """
        hook = decomp.get("hook")
        fn = getattr(self, "hook_" + hook, None) if hook else None
        if fn:
            fn(ctx)
            return True
        return False

    # ------------------------------------------------------------- utilities
    def param(self, ctx, spec, default=None):
        """Resolve an eliza.json param spec against the matched context.

        int  -> '*' capture number (1-based) first, else Eliza COMP-style
                word position: pattern token N maps to sentence token N
                (wildcards consume their captured span). If the position is
                past the end, return the rest of the sentence.
        str  -> dotted lookup, e.g. 'user.location', or literal text.
        """
        if isinstance(spec, int):
            if 1 <= spec <= len(ctx["captures"]):
                return ctx["captures"][spec - 1]
            pos = self._token_at(ctx, spec)
            return pos if pos is not None else default
        if isinstance(spec, str) and "." in spec:
            head, rest = spec.split(".", 1)
            src = {"user": self.shell.state.user,
                   "session": self.shell.state.session,
                   "blackboard": self.shell.state.board}.get(head)
            if src is not None:
                cur = src
                for part in rest.split("."):
                    if isinstance(cur, dict):
                        cur = cur.get(part)
                    else:
                        return default
                return cur if cur is not None else default
        return spec

    @staticmethod
    def _token_at(ctx, n):
        """Eliza word-position: walk pattern tokens vs sentence tokens.

        Pattern '*' consumes all words up to the next literal pattern token;
        literal tokens consume one word each. Returns the sentence word at
        position n, or the remaining tail if n lands inside a trailing '*'.
        """
        pat = (ctx.get("decomp") or {}).get("pattern", "").lower().split()
        toks = ctx["tokens"]
        if not pat:
            return toks[n - 1] if 1 <= n <= len(toks) else None
        si = 0
        for pi, ptok in enumerate(pat):
            if si >= len(toks):
                return None
            if ptok == "*":
                nxt = pat[pi + 1] if pi + 1 < len(pat) else None
                if nxt is None:                      # trailing star: rest
                    return " ".join(toks[si:]) if n > si else \
                        toks[n - 1] if n >= 1 else None
                j = si
                while j < len(toks) and toks[j] != nxt:
                    j += 1
                if n <= j:
                    return toks[n - 1] if 1 <= n <= len(toks) else None
                si = j
            elif ptok.startswith("@"):
                syns = ctx.get("synonyms_by_name", {}).get(ptok[1:], [])
                matched = next((s for s in sorted(syns, key=len, reverse=True)
                                if " ".join(toks[si:si + len(s.split())]) == s),
                               None)
                if matched:
                    w = len(matched.split())
                    if n <= si + w - 1:
                        return toks[n - 1]
                    si += w
                else:
                    si += 1                          # best effort
            else:
                if n == si + 1:
                    return toks[si]
                si += 1
        return " ".join(toks[si:]) if si < len(toks) else None


def register(cls):
    REGISTRY.append(cls)
    return cls
