"""System plugins: salience engine, blackboard, metacognition, proactive.

These implement the 'sovereign logic' of eliza.json -- the parts that make
the loop feel alive without ever calling a model:

* SalienceEngine  topic scores that boost on keyword hit and decay per tick;
                  attention_focus on the blackboard = argmax salience.
* Blackboard      shared inferred state + named signals with priorities.
* Metacognition   loop detection over response_history (windowed repeats)
                  and context-loss checks -> breakout responses.
* Proactive       evaluates eliza.json trigger conditions against live
                  memory with a cooldown so it never nags.
"""
import re
import time
from elizaclaw.plugin import Plugin, register


@register
class SalienceEngine(Plugin):
    name = "salience_engine"
    priority = 98

    def update(self, topic, boost=None):
        cfg = self.shell.cfg["weights"]["salience"]
        st = self.shell.state.salience
        base = self.shell.cfg["weights"]["topics"].get(topic, {}) \
            .get("base", 3.0)
        cur = st["topics"].get(topic, {"score": 0.0})
        cur["score"] = min(cfg["max_score"],
                           (cur["score"] or base * 0.5) +
                           (boost if boost is not None
                            else cfg["boost_on_match"]))
        st["topics"][topic] = cur
        # --- semantic propagation (WordNet/ConceptNet-flavoured spreading
        # activation): related topics get a fraction of the boost, so talking
        # about 'work' keeps 'task' and 'project' warm in attention_focus.
        lex = getattr(self.shell, "lexicon", None)
        if lex and lex.available:
            try:
                for rel in lex.related(topic)[:8]:
                    w = rel["word"]
                    if w == topic.lower() or len(w.split()) > 2:
                        continue
                    kin = st["topics"].get(w)
                    if kin is not None:
                        kin["score"] = min(
                            cfg["max_score"],
                            kin["score"] + 0.3 * (boost or
                                                  cfg["boost_on_match"]))
                    elif rel["rel"] in ("IsA", "RelatedTo") \
                            and w in self.shell.cfg["weights"]["topics"]:
                        st["topics"][w] = {"score": base * 0.4}
            except Exception:
                pass
        self.shell.state.save("salience", st)

    def top(self):
        st = self.shell.state.salience
        topics = st["topics"]
        if not topics:
            return None
        return max(topics, key=lambda t: topics[t]["score"])

    def tick(self):
        cfgw = self.shell.cfg["weights"]["salience"]
        st = self.shell.state.salience
        now = time.time()
        since = now - st.get("last_decay", now)
        if since < 5:
            return
        for topic, cur in list(st["topics"].items()):
            rate = self.shell.cfg["weights"]["topics"] \
                .get(topic, {}).get("decay", cfgw["decay_rate"])
            cur["score"] = max(0.0, cur["score"] - rate * min(since, 60))
            if cur["score"] < cfgw["min_threshold"]:
                del st["topics"][topic]
        st["last_decay"] = now
        self.shell.state.save("salience", st)
        board = self.shell.state.board
        board["inferred_state"]["attention_focus"] = self.top()
        self.shell.state.save("blackboard", board)


@register
class BlackboardManager(Plugin):
    name = "blackboard_manager"
    priority = 99

    def emit(self, kind, data=None):
        spec = self.shell.cfg["blackboard"]["signals"].get(kind, {})
        board = self.shell.state.board
        board["signals"].append({"t": time.time(), "kind": kind,
                                 "priority": spec.get("priority", 5),
                                 "action": spec.get("action", "note")})
        del board["signals"][:-20]
        act = spec.get("action")
        inf = board["inferred_state"]
        if act == "focus":
            inf["urgency"] = "high"
        elif act == "shift_tone":
            inf["emotional_state"] = "frustrated"
        elif act == "breakout":
            inf["looping"] = True
        self.shell.state.save("blackboard", board)

    def hook_update_inference(self, ctx):
        """on_turn: fold the latest turn into the shared inferred state."""
        board = self.shell.state.board
        inf = board["inferred_state"]
        toks = set(ctx["tokens"])
        tech = toks & set(self.shell.cfg["synonyms"].get("code", [])) \
            | toks & {"api", "regex", "async", "deploy", "refactor"}
        if tech:
            inf["technical_level"] = "advanced" if "traceback" in toks \
                or "stack" in toks else "expert"
        if any(w in toks for w in ("deadline", "urgent", "asap", "today")):
            inf["urgency"] = "high"
        elif inf.get("urgency") == "high" and not \
                [s for s in board["signals"]
                 if time.time() - s["t"] < 120]:
            inf["urgency"] = "normal"
        inf["attention_focus"] = \
            self.shell.plugins["salience_engine"].top() or \
            inf.get("attention_focus")
        board["last_inference"] = time.time()
        self.shell.state.save("blackboard", board)


@register
class Metacognition(Plugin):
    name = "metacognition"
    priority = 97

    def hook_check_loop(self, ctx):
        laws = self.shell.cfg["sovereign_logic"]["loop_detection"]
        win = self.state.setdefault("recent", [])
        # track what we actually SAID -- repeating ourselves == the loop
        item = (ctx.get("reply") or "").strip().lower()
        if not item:
            return False
        win.append(item)
        del win[:-laws["window"]]
        counts = {}
        for entry in win:
            counts[entry] = counts.get(entry, 0) + 1
        if max(counts.values(), default=0) >= laws["trigger_threshold"]:
            self.shell.plugins["blackboard_manager"].emit("loop_detected")
            ctx["breakout"] = True
            return True
        return False

    def hook_check_context(self, ctx):
        # context lost == nothing on the blackboard and empty salience after
        # several turns of input
        sess = self.shell.state.session
        if sess["turns"] > 4 and not self.shell.state.salience["topics"]:
            ctx["context_lost"] = True
            return True
        return False

    def hook_breakout(self, ctx):
        ctx["response"] = "generic.breakout"


@register
class Proactive(Plugin):
    name = "proactive"
    priority = 50

    COND_RX = re.compile(r"^([\w_.]+)\s*(>|<|>=|<=|==|!=|contains)\s*"
                         r"(.+)$")

    def _lookup(self, dotted):
        cur = {"session": self.shell.state.session,
               "task_stack": self.shell.state.tasks,
               "blackboard": self.shell.state.board,
               "weather": self.shell.plugins["weather"].state,
               "user": self.shell.state.user}
        for part in dotted.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        return cur

    def evaluate(self):
        cfg = self.shell.cfg["proactive"]
        if not cfg.get("enabled"):
            return
        last = self.state.get("last_fire", 0)
        if time.time() - last < cfg.get("cooldown", 300):
            return
        for trig in cfg["triggers"]:
            ok = False
            cond = trig["condition"]
            for clause in re.split(r"\s+AND\s+", cond):
                m = self.COND_RX.match(clause.strip())
                if not m:
                    ok = False
                    break
                lhs, op, rhs = m.groups()
                val = self._lookup(lhs)
                rhs = rhs.strip("'\" ")
                try:
                    if op == ">":
                        ok = float(val or 0) > float(rhs)
                    elif op == "<":
                        ok = float(val or 0) < float(rhs)
                    elif op == "==":
                        ok = str(val) == rhs or _boolish(val, rhs)
                    elif op == "!=":
                        ok = str(val) != rhs
                    elif op == "contains":
                        ok = rhs.lower() in str(val or "").lower()
                    else:
                        ok = False
                except (TypeError, ValueError):
                    ok = False
                if not ok:
                    break
            if ok:
                name = trig["name"]
                fired = self.state.setdefault("fired", {})
                if fired.get(name, 0) == int(time.time() // 600):
                    continue                       # already fired this epoch
                fired[name] = int(time.time() // 600)
                self.state["last_fire"] = time.time()
                self.save()
                tmpl = {"name": self.shell.state.user.get("name") or "friend",
                        "location": self.shell.state.user.get("location"),
                        "task": (self.shell.state.tasks["stack"] or
                                 [{"task": "that"}])[0]["task"],
                        "topic": self._lookup(
                            "blackboard.inferred_state.attention_focus")
                            or "your interests"}
                msg = self.shell.render_list(trig["responses"], tmpl)
                self.shell.queue(msg)
                return                             # one proactive nudge max

    def tick(self):
        self.evaluate()


def _boolish(val, rhs):
    if isinstance(val, bool):
        return val == (rhs.lower() == "true")
    return False
