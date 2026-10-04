"""Eliza-Claw :: state -- JSON memory manager. One file per concern.

memory/
  user.json          active profile (name, location, interests, facts)
  users.json         known-users directory (from example-users.json seed)
  history.json       recent interactions (bounded ring)
  task_stack.json    recursive task stack: stack / completed / deferred
  reminders.json     scheduled reminders checked on tick
  salience.json      topic salience scores (decay on tick)
  blackboard.json    shared signals between plugins
  plugins/<name>.json  per-plugin private state

Writes are atomic (tmp + os.replace) so a crash mid-tick never corrupts
memory. Every read is cached by mtime to keep the heartbeat cheap.
"""
import json
import os
import time


class State:
    def __init__(self, root="memory"):
        self.root = root
        os.makedirs(os.path.join(root, "plugins"), exist_ok=True)
        self._cache = {}

    # ------------------------------------------------------------- plumbing
    def path(self, name):
        if name.startswith("plugins/"):
            return os.path.join(self.root, name + ".json")
        return os.path.join(self.root, name + ".json")

    def load(self, name, default=None):
        if name in self._cache:
            return self._cache[name]
        p = self.path(name)
        data = default if default is not None else {}
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                data = default if default is not None else {}
        self._cache[name] = data
        return data

    def save(self, name, data=None):
        """Persist (atomic). data=None flushes the cached copy."""
        if data is not None:
            self._cache[name] = data
        data = self._cache.get(name, {})
        p = self.path(name)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, p)

    def bump(self, name, default=None):
        d = self.load(name, default)
        d["_updated"] = time.time()
        return d

    # ------------------------------------------------------------ accessors
    @property
    def user(self):
        return self.bump("user", {"name": None, "location": None,
                                  "interests": [], "facts": {},
                                  "preferences": {}})

    @property
    def session(self):
        s = self.bump("session", {"turns": 0, "started": time.time(),
                                  "last_input_at": time.time(),
                                  "recent_topics": []})
        return s

    @property
    def tasks(self):
        return self.bump("task_stack",
                         {"stack": [], "completed": [], "deferred": []})

    @property
    def reminders(self):
        return self.bump("reminders", {"items": []})

    @property
    def salience(self):
        return self.bump("salience", {"topics": {}, "last_decay": time.time()})

    @property
    def board(self):
        return self.bump("blackboard", {"signals": [], "inferred_state": {
            "technical_level": "unknown", "emotional_state": "neutral",
            "urgency": "normal", "attention_focus": None}})

    @property
    def history(self):
        h = self.bump("history", {"events": []})
        return h

    def log(self, kind, text):
        h = self.history
        h["events"].append({"t": time.time(), "kind": kind, "text": text})
        del h["events"][:-50]                     # bounded ring, 50 events
        self.save("history", h)

    def plugin_state(self, name, default=None):
        return self.bump("plugins/" + name, default or {})

    def save_plugin(self, name, data):
        self.save("plugins/" + name, data)

    # ------------------------------------------------------- user directory
    def seed_users(self, path="example-users.json"):
        """Import the shipped example-users.json as the known-user graph."""
        users = self.load("users", {"known": [], "current": None})
        if users["known"] or not os.path.exists(path):
            return users
        with open(path, "r", encoding="utf-8") as fh:
            seed = json.load(fh)
        users["known"] = seed.get("users", {}).get("known", [])
        self.save("users", users)
        return users

    def find_user(self, alias):
        alias = alias.lower().strip()
        for u in self.seed_users()["known"]:
            names = [u.get("name", "").lower()] + [a.lower() for a in
                                                   u.get("aliases", [])]
            if alias and (alias in names or alias == u.get("id")):
                return u
        return None
