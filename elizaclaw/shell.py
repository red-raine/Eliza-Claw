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
from elizaclaw.lexicon import Lexicon
from elizaclaw.plugin import REGISTRY
from elizaclaw.state import State
from elizaclaw.wordweb import WordWeb


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
        # --- Wave 1-4 brain modules (all optional, all degrade gracefully)
        self.lexicon = Lexicon(root.strip("/"))
        self.wordweb = WordWeb(root=self.cfg["config"].get(
            "wordweb_path", "data"), lex=self.lexicon)
        self.wordweb.load()                      # silent if not built yet
        self.turn = 0
        self._outbox = []
        self._compiled = {}          # pattern -> regex cache
        self.plugins = {}
        self._load_plugins()
        self._expand_synonyms()
        self._load_clan_scripts()
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

    # ------------------------------------------------- wave 0: JSON hygiene
    BUILTIN_KEYS = {
        # Safety net for eliza.json drift: every intent classifier category
        # must be reachable. If the script author forgot a key, we inject a
        # minimal one here (Weizenbaum's xnone trick generalized: the shell
        # guarantees coverage, the JSON only customizes it).
        "name": {"keyword": "name", "priority": 25, "plugin": "identity",
                 "synonyms": ["call", "called", "who"],
                 "decomps": [{"pattern": "* your name", "hook": "identify",
                              "response": "auto"},
                             {"pattern": "* name is *", "hook": "identify",
                              "response": "auto"}]},
        "task": {"keyword": "task", "priority": 12, "plugin": "task_manager",
                 "synonyms": ["todo", "to-do", "add task"],
                 "decomps": [{"pattern": "* add task *", "hook": "push_task",
                              "params": {"task": "cap1"},
                              "response": "task.acknowledged"},
                             {"pattern": "* new task *", "hook": "push_task",
                              "params": {"task": "cap1"},
                              "response": "task.acknowledged"},
                             {"pattern": "* tasks", "hook": "check_stack",
                              "response": "auto"}]},
        "help": {"keyword": "help", "priority": 22, "plugin": None,
                 "synonyms": [],
                 "decomps": [{"pattern": "* help *", "response":
                              "generic.help"}]},
        "thanks": {"keyword": "thanks", "priority": 6, "plugin": None,
                   "synonyms": ["thank"],
                   "decomps": [{"pattern": "*", "response":
                                "generic.thanks"}]},
        "goodbye": {"keyword": "goodbye", "priority": 6, "plugin": None,
                    "synonyms": ["bye", "see you"],
                    "decomps": [{"pattern": "*", "response":
                                 "generic.goodbye"}]},
    }

    def _build_keys(self):
        """Sort keys by priority and precompile every decomp pattern."""
        names = {k["keyword"] for k in self.cfg["keys"]}
        for nm, spec in self.BUILTIN_KEYS.items():
            if nm not in names:
                self.cfg["keys"].append(dict(spec))
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

    # -------------------------------------------------- wave 1: synonym growth
    def _expand_synonyms(self):
        """WordNet-grown @syn sets (memoized in memory/synsets_expanded.json).
        Eliza's DOCTOR script hand-wrote 'alike equal same identical'; we mine
        150k synsets so every key speaks in more than one dialect."""
        if not self.lexicon.has_wordnet:
            return
        cache = self.state.load("synsets_expanded", {})
        changed = False
        for name, words in list(self.cfg["synonyms"].items()):
            if name in cache:
                self.cfg["synonyms"][name] = cache[name]
                continue
            try:
                grown = self.lexicon.expand_synset(words)
            except Exception:
                grown = words
            cache[name] = grown
            self.cfg["synonyms"][name] = grown
            changed = True
        if changed:
            self.state.save("synsets_expanded", cache)

    # --------------------------------------------------- wave 3: clan scripts
    def _load_clan_scripts(self):
        from elizaclaw import clan
        folder = self.cfg["config"].get("scripts_path", "scripts")
        self._clan_loaded = clan.load_scripts(self, folder)

    def reload_scripts(self):
        self._compiled.clear()
        self._load_clan_scripts()
        self._build_keys()
        return getattr(self, "_clan_loaded", [])

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
        # Wave 4: the Bayesian word-web votes too (Pandemonium, Selfridge 1956:
        # let every specialist demon shout, then take the loudest consensus).
        web_route, web_conf, web_tab = (None, 0.0, {})
        if self.wordweb.w3:
            try:
                web_route, web_conf, web_tab = self.wordweb.route(sentence)
            except Exception:
                pass
        ctx = {"sentence": sentence, "tokens": tokens, "intent": intent,
               "captures": [], "vars": {}, "decomp": {}, "turn": self.turn,
               "response": None, "breakout": False, "context_lost": False,
               "web_route": web_route, "web_conf": web_conf,
               "web_scores": web_tab}

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

    ROUTE2KEY = {"task": "task", "reminder": "remind", "info": None,
                 "chat": None, "system": None}

    def _key_for_route(self, want, joined):
        """Wave 0 completion: find the best pattern match belonging to the
        plugin/key the Bayesian word-web wants ('task', 'remind', ...).
        Returns (rank_tuple, key, decomp, match, ngroups) or None."""
        cand = None
        for key in self.keys:
            if key.get("plugin") != want and key["keyword"] != want:
                continue
            prio = key.get("priority", 0)
            for decomp in key["decomps"]:
                fn, ng = self._compiled[decomp["pattern"].lower()]
                caps = fn(joined)
                if caps is not None:
                    rank = (-prio, -self._specificity(decomp["pattern"]))
                    if cand is None or rank < cand[0]:
                        cand = (rank, key, decomp, caps, ng)
                    break
        return cand

    def _web_fallback(self, ctx):
        """No pattern matched at all.  Two eras of ideas combine here:
        1. Weizenbaum's xnone -- ask the user to rephrase (cheap, polite).
        2. Selfridge's Pandemonium -- if the Bayesian word-web shouted with
           high confidence, try its preferred route's generic opener anyway
           so a statistically-obvious request isn't bounced.
        Otherwise fall through to the xnone key's responses."""
        wr = ctx.get("web_route")
        if wr and ctx.get("web_conf", 0) >= 0.65:
            alt = self._key_for_route(self.ROUTE2KEY.get(wr) or wr,
                                      " ".join(ctx["tokens"]))
            if alt:
                return self._dispatch(alt, ctx)
        xn = next((k for k in self.keys if k["keyword"] == "xnone"), None)
        if xn:
            fn, ng = self._compiled[xn["decomps"][0]["pattern"].lower()]
            caps = fn(" ".join(ctx["tokens"]))
            if caps is not None:
                return self._dispatch(((0, 0), xn, xn["decomps"][0],
                                       caps, ng), ctx)
        return False                                 # nothing left to try

    def _dispatch(self, best, ctx):
        _, key, decomp, caps, ng = best
        ctx["key"] = key
        ctx["decomp"] = decomp
        ctx["captures"] = [c for c in caps[:ng] if c != ""] or []
        return False

    def _best_match(self, joined, only=None):
        """Scan all keys (or just plugin/key == `only`) for the highest-ranked
        decomp pattern match. Returns (rank, key, decomp, caps, ng) or None."""
        best = None
        for key in self.keys:
            if only and key.get("plugin") != only and key["keyword"] != only:
                continue
            if key["_hits"] and not any(rx.search(joined)
                                        for rx in key["_rxs"]):
                continue                             # cheap keyword gate
            prio = key.get("priority", 0)
            for decomp in key["decomps"]:
                fn, ng = self._compiled[decomp["pattern"].lower()]
                caps = fn(joined)
                if caps is not None:
                    rank = (-prio, -self._specificity(decomp["pattern"]))
                    if best is None or rank < best[0]:
                        best = (rank, key, decomp, caps, ng)
                    break                            # within-key, first hit
        return best

    def route(self, ctx):
        joined = " ".join(ctx["tokens"])
        best = self._best_match(joined)
        if not best:
            return self._web_fallback(ctx)
        # Word-web arbitration (Wave 4): the Bayesian router casts a vote.
        # Strong statistical evidence outranks a weak pattern match -- but
        # never overrides an explicit high-priority keyword rule.
        wr = ctx.get("web_route")
        prio = best[1].get("priority", 0)
        if wr and ctx.get("web_conf", 0) >= 0.5 and prio <= 10:
            matched_key = best[1]["keyword"]
            want = self.ROUTE2KEY.get(wr)
            if want and want != matched_key and want != "xnone":
                alt = self._key_for_route(want, joined)
                if alt and -alt[0][0] >= -prio:      # alt priority >= ours
                    best = alt
        if not best:
            return self._web_fallback(ctx)
        _, key, decomp, caps, ng = best
        ctx["key"] = key
        ctx["decomp"] = decomp
        # Wave 0 (bug #22): a pattern may legitimately bind an empty capture
        # ('* remind me *' on 'remind me about the taxes later' -> cap1='').
        # Keep every capture in positional order and remember which slots
        # were blank; plugins that want only non-empty spans call clean().
        ctx["captures"] = list(caps[:ng])
        ctx["empty_caps"] = [i for i, c in enumerate(ctx["captures"])
                             if c == ""]

        # ---- intent-driven task auto-capture (ELIZA's 'need' key, promoted
        # to a general reflex). If nothing more specific claimed this turn as
        # an action, but the sentence is a request ('can you ...', 'please
        # ...', imperative opener), push it onto the task stack. This is the
        # zero-token answer to "agent memory": we simply write down what was
        # asked, exactly like Colbys object-assumption frames fill slots.
        if (key["keyword"] == "xnone" and ctx.get("intent") == "command"
                and len(ctx["tokens"]) >= 3):
            tm = self.plugins.get("task_manager")
            if tm:
                tctx = dict(ctx)
                tctx["captures"] = [" ".join(ctx["tokens"])]
                tctx["decomp"] = {"pattern": "*"}
                if tm.hook_push_task(tctx) and tctx.get("response"):
                    ctx["vars"].update(tctx["vars"])
                    ctx["response"] = tctx["response"]

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
