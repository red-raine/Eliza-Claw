"""Task manager plugin: the recursive task stack. push / pop / defer / review.

This is the 'agent' part of the micro-OS -- but notice it is just a list in
a JSON file plus four functions. The heartbeat reviews it; metacognition
checks for loops. No model, no tokens, fully inspectable.
"""
import time
from elizaclaw.plugin import Plugin, register


@register
class TaskManager(Plugin):
    name = "task_manager"
    priority = 90

    # ------------------------------------------------------------- hooks
    def hook_push_task(self, ctx):
        spec = ctx["decomp"].get("params", {})
        raw = self.param(ctx, spec.get("task", 2))
        if not raw:
            return False
        task = {"task": str(raw), "priority": spec.get("priority", "medium"),
                "pushed": time.time(), "turn": ctx["turn"]}
        st = self.shell.state.tasks
        st["stack"] = [t for t in st["stack"] if t["task"] != task["task"]]
        # priority insert: high items bubble to top of the working stack
        rank = {"high": 0, "medium": 1, "low": 2}
        st["stack"].append(task)
        st["stack"].sort(key=lambda t: (rank.get(t["priority"], 1),
                                        -t["pushed"]))
        del st["stack"][10:]                       # max_depth from eliza.json
        self.shell.state.save("task_stack", st)
        ctx["vars"]["task"] = task["task"]
        self.signal("task_added")

    def hook_pop_task(self, ctx):
        raw = self.param(ctx, ctx["decomp"].get("params", {}).get("task", 2))
        st = self.shell.state.tasks
        target = None
        if raw:
            raw_l = str(raw).lower()
            for t in st["stack"]:
                if raw_l in t["task"].lower() or t["task"].lower() in raw_l:
                    target = t
                    break
        if target is None and st["stack"]:
            target = st["stack"][0]
        if target:
            st["stack"].remove(target)
            st["completed"].append(target["task"])
            del st["completed"][:-20]
            self.shell.state.save("task_stack", st)
            ctx["vars"]["task"] = target["task"]
            self.signal("task_completed")
            u = self.shell.state.user
            u.setdefault("history", {})["tasks_completed"] = \
                u.get("history", {}).get("tasks_completed", []) + \
                [target["task"]]
            self.shell.state.save("user", u)
            return True
        ctx["vars"]["task"] = raw or "that"
        return True

    def hook_defer_task(self, ctx):
        st = self.shell.state.tasks
        if st["stack"]:
            t = st["stack"].pop(0)
            st["deferred"].append(t)
            self.shell.state.save("task_stack", st)
            ctx["vars"]["task"] = t["task"]

    def hook_check_stack(self, ctx=None):
        """Heartbeat action: nudge about the oldest high-priority item."""
        st = self.shell.state.tasks
        due = [t for t in st["stack"]
               if t["priority"] == "high" and time.time() - t["pushed"] > 300]
        if due and ctx is not None:
            ctx["vars"]["task"] = due[0]["task"]
            ctx["response"] = "task.progress_check"
            return True
        return False

    # ------------------------------------------------------------ signals
    def signal(self, kind):
        board = self.shell.state.board
        board["signals"].append({"t": time.time(), "kind": kind})
        del board["signals"][:-20]
        if kind == "task_completed":
            board["inferred_state"]["urgency"] = "normal"
        self.shell.state.save("blackboard", board)

    # ----------------------------------------------------------- heartbeat
    def tick(self):
        ctx = {"vars": {}, "captures": [], "tokens": [], "decomp": {},
               "turn": self.shell.turn}
        if self.hook_check_stack(ctx) and ctx.get("response"):
            msg = self.shell.render(ctx["response"], ctx["vars"])
            if msg:
                self.shell.queue(msg)
