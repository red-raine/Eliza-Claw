"""Eliza-Claw :: plugin_clan -- executes compiled .clan scripts.

Registered as plugin 'clan' so eliza.json keys with plugin:"clan" route here.
Actions run in order; a reply action wins unless a hook already produced a
response. $n refers to capture n, {var} resolves through the mad-lib engine
(user/session/vars), keeping scripts and JSON templates interoperable.
"""
import re

from elizaclaw.plugin import Plugin, register


@register
class ClanPlugin(Plugin):
    name = "clan"
    priority = 5

    def handle(self, key, decomp, ctx):
        spec = key.get("clan") or {}
        actions = spec.get("actions", [])
        reply = None
        for act in actions:
            kind = act["kind"]
            if kind == "hook":
                self._run_hook(act["args"], ctx)
            elif kind == "shell":
                self.shell.run_action(act["path"], ctx)
            elif kind == "calc":
                try:
                    val = eval(act["expr"], {"__builtins__": {}}, {})  # noqa
                    ctx["vars"]["result"] = val
                except Exception:
                    ctx["vars"]["result"] = "?"
            elif kind == "reply":
                reply = act["template"]
        if reply is not None and ctx.get("response") is None:
            ctx["response"] = self._fill(reply, ctx)
            ctx["clan_direct"] = True
        return bool(actions)

    def _run_hook(self, args, ctx):
        """key=value pairs -> first key names the plugin, rest are params."""
        if not args:
            return
        items = list(args.items())
        target, field = items[0]
        plugin = self.shell.plugins.get(target)
        if plugin is None:
            return
        params = {}
        for k, v in items[1:]:
            params[k] = self._sub(v, ctx)
        hook = getattr(plugin, "hook_" + field.replace("-", "_"), None) \
            if field.replace("-", "_") != field or True else None
        # field may be 'add'/'remove' etc.; also accept plain param pass
        if hook:
            ctx2 = dict(ctx)
            ctx2["vars"] = dict(ctx.get("vars", {}))
            ctx2["vars"].update(params)
            try:
                hook(ctx2)
                if ctx2.get("response") and not ctx.get("response"):
                    ctx["response"] = ctx2["response"]
            except Exception:
                pass

    def _sub(self, s, ctx):
        def dollar(m):
            n = int(m.group(1))
            caps = ctx.get("captures") or []
            return caps[n - 1] if 1 <= n <= len(caps) else ""
        s = re.sub(r"\$(\d+)", dollar, str(s))
        return s

    def _fill(self, tmpl, ctx):
        tmpl = self._sub(tmpl, ctx)
        if re.search(r"\{|\$", tmpl) or '"' in tmpl:
            pool = [tmpl]
            return self.shell.render_list(pool, ctx.get("vars", {}))
        return tmpl
