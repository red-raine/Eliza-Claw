"""Eliza-Claw :: shell -- the microkernel. load -> parse -> route ->
execute -> template -> respond, wrapped in a heartbeat.

Everything the plugins need lives here:
  * config loaded from eliza.json (the single source of truth)
  * plugin registry auto-imported from elizaclaw/plugins/
  * the router: keyword+synonym match, decomp pattern match, priority order
  * the renderer: mad-lib templates with entropy + repeat avoidance
  * the agents from eliza.json ('heartbeat' etc.) driven by on_tick/on_turn
"""
import importlib
import os
import pkgutil
import random
import re
import time

from elizaclaw import nlu, plugins as _plugins
from elizaclaw.plugin import REGISTRY
from elizaclaw.state import State


def _kw_rx(h):
    """Whole-word keyword matcher (compiled once per hit string)."""
    return re.compile(r"(?<!\w)%s(?!\w)" % re.escape(h), re.I)


class Shell:
    def __init__(self, config_path="eliza.json", memory_root=None):
        import json
        with open(config_path, "r", encoding="utf-8") as fh:
            self.cfg = json.load(fh)
        root = memory_root or self.cfg["config"].get("memory_path", "./memory/")
        self.state = State(root.strip("/"))
        self.turn = 0
        self._outbox = []
        self._compiled = {}          # pattern -> regex cache
        self.plugins = {}
        self._load_plugins()
        self._build_keys()

    # ----------------------------------------------------------- bootstrap
    def _load_plugins(self):
        for mod in pkgutil.iter_modules(_plugins.__path__):
            importlib.import_module("elizaclaw.plugins." + mod.name)
        spec = {p["name"]: p for p in self.cfg.get("plugins", [])}
        for cls in sorted(REGISTRY, key=lambda c: -getattr(c, "priority", 0)):
            if spec.get(cls.name, {}).get("enabled") is False:
                continue                            # eliza.json disables it
            inst = cls(self)
            self.plugins[inst.name] = inst

    def _build_keys(self):
        """Sort keys by priority and precompile every decomp pattern."""
        self.keys = sorted(self.cfg["keys"],
                           key=lambda k: -k.get("priority", 0))
        syn = self.cfg["synonyms"]
        for key in self.keys:
            hits = [key["keyword"]] + key.get("synonyms", [])
            key["_hits"] = [h for h in hits if h != "xnone"]
            key["_rxs"] = [_kw_rx(h) for h in key["_hits"]]
            for decomp in key["decomps"]:
                pat = decomp["pattern"].lower()
                if pat not in self._compiled:
                    self._compiled[pat] = nlu.compile_pattern(pat, syn)

    # ------------------------------------------------------------ pipeline
    def step(self, text):
        """One full turn: load->parse->route->execute->template->respond."""
        self.turn += 1
        sess = self.state.session
        sess["turns"] = self.turn
        sess["last_input_at"] = time.time()
        self.state.save("session", sess)
        self.state.log("user", text)

        sentence = nlu.apply_pre(text, self.cfg["transformers"]["pre"]).lower()
        tokens = nlu.tokenize(sentence)
        intent = nlu.classify_intent(sentence,
                                     self.cfg["intent_classifier"],
                                     self.cfg["synonyms"])
        ctx = {"sentence": sentence, "tokens": tokens, "intent": intent,
               "captures": [], "vars": {}, "decomp": {}, "turn": self.turn,
               "response": None, "breakout": False, "context_lost": False}

        exit_now = self.route(ctx)
        # on_turn agents from eliza.json (metacognition, inference engine)
        for agent in self.cfg.get("agents", []):
            if agent.get("trigger") == "on_turn" and agent.get("enabled"):
                for action in agent["actions"]:
                    self.run_action(action, ctx)

        reply = self.render_ctx(ctx)
        # metacognition sees the ACTUAL outgoing reply; a loop it detects
        # surfaces on the next beat via the heartbeat queue.
        if self.plugins["metacognition"].hook_check_loop({"reply": reply}):
            msg = self.render("generic.breakout", ctx["vars"])
            if msg:
                self.queue(msg)
        if reply:
            self.state.log("bot", reply)
        if exit_now:
            return self.cfg["config"]["final"], True
        queued = self.flush_queue()
        out = " ".join(x for x in [reply] + queued if x).strip()
        return out or self.render("generic.continue", {}), False

    # --------------------------------------------------------------- routing
    @staticmethod
    def _specificity(pattern):
        """More literal tokens beat fewer; '*' is free. Classic Eliza order."""
        toks = pattern.split()
        return sum(1 for t in toks if t != "*") * 10 + len(toks)

    def route(self, ctx):
        joined = " ".join(ctx["tokens"])
        best = None                                  # (rank, key, decomp, m)
        for key in self.keys:
            if key["_hits"] and not any(rx.search(joined)
                                        for rx in key["_rxs"]):
                continue                             # cheap keyword gate
            prio = key.get("priority", 0)
            for decomp in key["decomps"]:
                rx, ng = self._compiled[decomp["pattern"].lower()]
                m = rx.match(joined)
                if m:
                    rank = (-prio, -self._specificity(decomp["pattern"]))
                    if best is None or rank < best[0]:
                        best = (rank, key, decomp, m, ng)
                    break                            # within-key, first hit
        if not best:
            return False
        _, key, decomp, m, ng = best
        ctx["key"] = key
        ctx["decomp"] = decomp
        ctx["captures"] = [g for g in m.groups() if g is not None]

        # salience boost + blackboard signal straight from the JSON spec
        topic = key["keyword"] if key["keyword"] != "xnone" else \
            (ctx["captures"][0].split()[0] if ctx["captures"] else None)
        if topic:
            self.plugins["salience_engine"].update(
                topic, key.get("salience_boost"))
            sess = self.state.session
            rt = sess["recent_topics"]
            if topic not in rt:
                rt.append(topic)
                del rt[:-8]
            self.state.save("session", sess)

        if decomp.get("action") == "exit":
            return True

        plugin_name = key.get("plugin", "eliza_conversation")
        plugin = self.plugins.get(plugin_name,
                                  self.plugins["eliza_conversation"])
        fired = plugin.handle(key, decomp, ctx)

        # response selection per JSON spec
        resp = decomp.get("response", "")
        if resp == "auto":
            pass                                   # hook chose ctx['response']
        elif resp.startswith(("generic.", "identity.", "task.", "emotion.",
                              "code.", "weather.", "search.", "pos.")):
            ctx.setdefault("explicit_response", resp)
            if ctx.get("response") is None:
                ctx["response"] = resp
        elif resp:
            head = plugin_name if plugin_name not in ("task_manager",
                                                      "eliza_conversation",
                                                      "identity") else "generic"
            ctx.setdefault("explicit_response", "%s.%s" % (head, resp))
            if ctx.get("response") is None:
                ctx["response"] = "%s.%s" % (head, resp)

        # 'learn' directive -> eliza_conversation stores the revealed fact
        learn = decomp.get("learn")
        if learn:
            self.plugins["eliza_conversation"].learn_fact(
                ctx, learn.get("category", "general"))
        sig = decomp.get("blackboard_signal")
        if sig:
            self.plugins["blackboard_manager"].emit(sig)
        return False

    # ------------------------------------------------------------ rendering
    def resolve_tree(self, dotted):
        node = self.cfg["responses"]
        for part in dotted.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            else:
                return None
        if node is None and dotted.startswith("pos."):
            pool = self.cfg.get("pos_transformers", {}).get(
                "reflection_templates", {})
            node = pool.get(dotted.split(".", 1)[1])
        if node is None and dotted.startswith("config."):
            node = self.cfg["config"].get(dotted.split(".", 1)[1])
        return node

    def render(self, dotted, variables):
        pool = self.resolve_tree(dotted) if dotted else None
        if isinstance(pool, str):
            pool = [pool]
        if not pool:
            return None
        return self.render_list(pool, variables)

    def render_list(self, pool, variables):
        """Mad-lib fill + entropy pick + avoid_repeat_last from settings."""
        rs = self.cfg["response_settings"]
        hist = self.state.load("render_hist", {"used": []})
        avoid = rs.get("avoid_repeat_last", 3)
        recent = hist["used"][-avoid:]
        choices = [t for t in pool if t not in recent] or pool
        if rs.get("mode") == "varied" and len(choices) > 1:
            tmpl = random.choice(choices)
        else:
            tmpl = choices[0]
        hist["used"].append(tmpl)
        del hist["used"][:-12]
        self.state.save("render_hist", hist)
        def sub(m):
            key = m.group(1)
            val = variables.get(key)
            return str(val) if val is not None else ""
        out = re.sub(r"\{(\w+)\}", sub, tmpl)
        # tidy clauses left dangling by missing vars
        out = re.sub(r"\s+(of|about|for|to|in)\s+([.,!?])",
                     lambda m: " " + m.group(1) + " it" + m.group(2), out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        out = re.sub(r"[.,;:]\s*[.,;:]", lambda m: m.group(0)[-1], out)
        return out

    def render_ctx(self, ctx):
        dotted = ctx.get("response") or ctx.get("explicit_response")
        variables = dict(ctx["vars"])
        variables.setdefault("name", self.state.user.get("name") or "friend")
        if dotted:
            msg = self.render(dotted, variables)
            if msg:
                return msg
        return self.render("generic.confused", variables)

    # ------------------------------------------------------------- heartbeat
    def queue(self, message):
        if message:
            self._outbox.append(message)

    def flush_queue(self):
        out, self._outbox = self._outbox, []
        return out

    def tick(self):
        """One heartbeat: run enabled on_tick agents at their intervals."""
        now = time.time()
        for agent in self.cfg.get("agents", []):
            if not agent.get("enabled") or agent.get("trigger") != "on_tick":
                continue
            interval = agent.get("interval", 1.0)
            last = self.state.load("ticks", {})
            key = "agent:" + agent["name"]
            if now - last.get(key, 0) < interval:
                continue
            last[key] = now
            self.state.save("ticks", last)
            for action in agent["actions"]:
                self.run_action(action, {"vars": {}, "tokens": [],
                                         "sentence": "", "turn": self.turn,
                                         "captures": [], "decomp": {}})
        for plugin in self.plugins.values():
            tk = "tick:" + plugin.name
            ivl = {"reminder": 5, "proactive": 10,
                   "task_manager": 60, "salience_engine": 5}.get(
                       plugin.name, 30)
            last = self.state.load("ticks", {})
            if now - last.get(tk, 0) >= ivl:
                last[tk] = now
                self.state.save("ticks", last)
                plugin.tick()
        return self.flush_queue()

    def run_action(self, dotted, ctx):
        plugin_fn = dotted.split(".")
        plugin = self.plugins.get(plugin_fn[0])
        if not plugin:
            return
        fn = getattr(plugin, "hook_" + "_".join(plugin_fn[1:]), None)
        if fn:
            try:
                fn(ctx)
            except Exception:
                pass                                # heartbeat must never die
