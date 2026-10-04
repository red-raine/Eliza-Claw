"""Eliza-Claw :: plugins.task_manager -- recursive task stack plugin.

Demonstrates that a 'complex' feature (a priority stack with deadlines) is
just a JSON file plus four functions. The heartbeat reviews it; metacognition
checks for loops. No model, no tokens, fully inspectable.

Vocabulary upgrade: task matching is no longer literal substring only. The
manager scores every pending item with a blend of token overlap, WordNet
Wu-Palmer similarity between head nouns, and ConceptNet RelatedTo edges --
so 'i finished the assignment' closes 'write the report', and near-synonym
re-adds dedupe into one entry instead of stacking clones. push also runs
FrameNet-lite slot filling (verb -> frame -> Theme/Time FEs via POS tags +
preposition cues) to extract clean task titles and deadlines.
"""
import time
from elizaclaw import nlu
from elizaclaw.plugin import Plugin, register


@register
class TaskManager(Plugin):
    name = "task_manager"
    priority = 90

    # ------------------------------------------------------ semantic scoring
    def _match_score(self, raw_l, title_l):
        """0..1 confidence that phrase `raw_l` refers to stack item `title_l`.

        Blend: literal containment (1.0), shared-token overlap, then
        lexical-semantic evidence from the shell's Lexicon (WordNet graph
        measures + mined ConceptNet edges). Max signal wins.
        """
        if not raw_l or not title_l:
            return 0.0
        if raw_l in title_l or title_l in raw_l:
            return 1.0
        shared = set(raw_l.split()) & set(title_l.split())
        lex = getattr(self.shell, "lexicon", None)
        if not (lex and lex.available):
            return min(0.9, 0.5 * len(shared))
        try:
            rw = lex.lemmatize(raw_l.split()[-1])
            tw = lex.lemmatize(title_l.split()[-1])
            wup = lex.similarity(rw, tw, "wup")
            path = lex.similarity(rw, tw, "path")
            rels = {r["word"] for r in lex.related(tw)}
            bridge = 0.75 if rw in rels else 0.0
            syn_overlap = 0.6 * min(1.0, len(shared) / 2.0)
            return max(wup, path * 0.9, bridge, syn_overlap)
        except Exception:
            return 0.0

    def _best_match(self, raw_l, stack, threshold=0.65):
        best, score = None, 0.0
        for t in stack:
            s = self._match_score(raw_l, t["task"].lower())
            if s > score:
                best, score = t, s
        return (best, score) if score >= threshold else (None, score)

    # ------------------------------------------------------------- hooks
    # priority words -> int (eliza.json may say "high"/"medium"/"low")
    PRIO_WORDS = {"critical": 5, "urgent": 5, "highest": 5, "asap": 5,
                  "high": 4, "hot": 4,
                  "medium": 3, "med": 3, "normal": 3, "mid": 3,
                  "low": 2, "someday": 1, "later": 1, "whenever": 1}

    def _prio(self, spec, ctx):
        """Resolve a priority spec to an int 1..5. Accepts ints, numeric
        strings, and English ('high', 'medium'). Scientific method: the demo
        crashed with int('medium'); now English is a first-class citizen."""
        raw = spec
        if isinstance(spec, str) and not spec.isdigit():
            # dotted lookup or literal word from eliza.json
            try:
                raw = self.param(ctx, spec)
            except Exception:
                raw = spec
        if isinstance(raw, (int, float)):
            return max(1, min(5, int(raw)))
        s = str(raw).strip().lower()
        if s.isdigit():
            return max(1, min(5, int(s)))
        return self.PRIO_WORDS.get(s, 3)

    def hook_push_task(self, ctx):
        spec = ctx["decomp"].get("params", {})
        raw = self.param(ctx, spec.get("task", 2))
        if not raw:
            return False

        # --- FrameNet-lite slot filling: verbs carry their own arguments ----
        # 'schedule dentist for friday' -> Appointment{Theme: dentist}.
        # No parser model here: POS tags + a preposition-cue table
        # (memory/frames.json) get us most of the way.
        due = parse_due(spec.get("due")) or nlu.parse_time(str(raw))
        title = str(raw)
        frame_info = None
        lex = getattr(self.shell, "lexicon", None)
        if lex and lex.available:
            try:
                toks = nlu.tokenize(ctx["sentence"])
                tagged = lex.tag(toks)
                frame_info = lex.fill_frame(toks, tagged)
                if frame_info:
                    slots = frame_info["slots"]
                    theme = slots.get("Theme") or slots.get("Patient")
                    if theme and 0 < len(theme.split()) <= 8 \
                            and not any(w in title.lower()
                                         for w in theme.lower().split()):
                        pass                     # glob already has the words
                    elif theme:
                        title = theme
                    when_txt = slots.get("Time") or slots.get("Duration")
                    if when_txt and due is None:
                        due = nlu.parse_time(when_txt)
            except Exception:
                frame_info = None

        st = self.shell.state.tasks
        entry = {"task": title,
                 "priority": self._prio(spec.get("priority", 3), ctx),
                 "added": time.time(),
                 "deadline": due}
        if frame_info:
            entry["frame"] = frame_info["frame"]
            entry["slots"] = frame_info["slots"]
        # semantic dedupe: 'the report' vs 'that report' == one task
        for t in st["stack"]:
            if self._match_score(title.lower(), t["task"].lower()) >= 0.85:
                t["priority"] = max(t.get("priority", 3), entry["priority"])
                target = t
                break
        else:
            st["stack"].append(entry)
            target = entry
        st["stack"].sort(key=lambda t: -t.get("priority", 3))
        self.shell.state.save("task_stack", st)
        ctx["vars"]["task"] = target["task"]
        ctx["response"] = "task.acknowledged"
        # salience: attention_focus should follow the newest task
        self.shell.plugins["salience_engine"].update(target["task"], boost=4)
        board = self.shell.state.board
        board["inferred_state"]["urgency"] = \
            "high" if (due and due - time.time() < 3600) \
            else board["inferred_state"].get("urgency", "normal")
        self.shell.state.save("blackboard", board)
        return True

    def hook_pop_task(self, ctx):
        raw = self.param(ctx, ctx["decomp"].get("params", {}).get("task", 2))
        st = self.shell.state.tasks
        target = None
        if raw:
            target, _score = self._best_match(str(raw).lower(), st["stack"])
        if target is None and st["stack"]:
            target = st["stack"][0]
        if target:
            st["stack"].remove(target)
            st["completed"].append({**target, "done_at": time.time()})
            del st["completed"][:-30]
            self.shell.state.save("task_stack", st)
            ctx["vars"]["task"] = target["task"]
            ctx["response"] = "task.completed"
            self.shell.plugins["blackboard_manager"].emit(
                "task_completed", {"task": target["task"]})
        return bool(target)

    def hook_defer_task(self, ctx):
        raw = self.param(ctx, ctx["decomp"].get("params", {}).get("task", 2))
        st = self.shell.state.tasks
        target = None
        if raw:
            target, _ = self._best_match(str(raw).lower(), st["stack"])
        if target is None and st["stack"]:
            target = st["stack"][0]
        if target:
            st["stack"].remove(target)
            st["deferred"].append({**target, "deferred_at": time.time()})
            del st["deferred"][:-30]
            self.shell.state.save("task_stack", st)
            ctx["vars"]["task"] = target["task"]
            ctx["response"] = "task.deferred"
        return bool(target)

    def hook_check_stack(self, ctx=None):
        st = self.shell.state.tasks
        now = time.time()
        overdue = [t for t in st["stack"]
                   if t.get("deadline") and t["deadline"] < now]
        top = st["stack"][0] if st["stack"] else None
        if ctx is not None and top:
            ctx["vars"]["task"] = top["task"]
            ctx["response"] = "task.progress_check"
            return True
        for t in overdue:
            self.shell.plugins["blackboard_manager"].emit(
                "deadline_breach", {"task": t["task"]})
        return bool(overdue)

    def tick(self):
        self.hook_check_stack()


def parse_due(value):
    """Deadline param may be a relative-seconds number or a natural phrase."""
    if value is None:
        return None
    try:
        return time.time() + float(value)
    except (TypeError, ValueError):
        pass
    return nlu.parse_time(str(value))
