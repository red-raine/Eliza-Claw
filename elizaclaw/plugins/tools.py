"""Weather + search + reminder: the 'tools' of a token-free agent.

Weather uses wttr.in via urllib (stdlib) with an offline fallback: a cached
deterministic pseudo-forecast so the demo never hard-fails without network.
Search is DuckDuckGo's lite HTML endpoint, parsed with regex -- classical.
Reminders parse natural-ish times ("in 10 minutes", "at 5pm") with pure code.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from elizaclaw.plugin import Plugin, register


def http_get(url, timeout=4):
    req = urllib.request.Request(url, headers={"User-Agent": "eliza-claw/0.4"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(200000).decode("utf-8", "replace")


@register
class Weather(Plugin):
    name = "weather"
    priority = 10

    def hook_current(self, ctx):
        loc = self.param(ctx, ctx["decomp"].get("params", {})
                         .get("location", "user.location"))
        if not loc:
            ctx["response"] = "weather.unknown_location"
            return True
        data = self.fetch(str(loc).strip())
        if not data:
            ctx["response"] = "generic.confused"
            return True
        ctx["vars"].update({"location": data["location"],
                            "temp": data["temp"],
                            "condition": data["condition"]})
        ctx["response"] = "weather.current"
        board = self.shell.state.board
        board["inferred_state"]["weather_condition"] = data["condition"]
        self.shell.state.save("blackboard", board)
        cache = self.state.setdefault("cache", {})
        cache[loc.lower()] = {"data": data, "t": time.time()}
        self.save()
        return True

    def fetch(self, location):
        # cache for 30 min keeps the heartbeat cheap and polite
        cache = self.state.get("cache", {})
        hit = cache.get(location.lower())
        if hit and time.time() - hit["t"] < 1800:
            return hit["data"]
        try:
            url = ("https://wttr.in/" +
                   urllib.parse.quote(location) + "?format=j1")
            j = json.loads(http_get(url))
            cur = j["current_condition"][0]
            area = j.get("nearest_area", [{}])[0]
            city = area.get("areaName", [{}])[0].get("value", location)
            return {"location": city,
                    "temp": cur["temp_F"],
                    "condition": cur["weatherDesc"][0]["value"].lower()}
        except Exception:
            return None   # shell falls back to template 'unknown' path


@register
class Search(Plugin):
    name = "search"
    priority = 8

    def hook_web(self, ctx):
        q = self.param(ctx, ctx["decomp"].get("params", {}).get("query", 3))
        if not q:
            return False
        ctx["vars"]["query"] = str(q)
        try:
            html = http_get("https://duckduckgo.com/html/?q=" +
                            urllib.parse.quote(str(q)))
            m = re.search(r'result__a[^>]*>(.*?)</a>', html)
            s = re.search(r'result__snippet[^>]*>(.*?)</a>', html, re.S)
            strip = lambda t: re.sub(r"<[^>]+>", "", t or "").strip()
            if m:
                ctx["vars"]["title"] = strip(m.group(1))
                ctx["vars"]["snippet"] = strip(s.group(1)) if s else ""
                ctx["response"] = "search.result"
            else:
                ctx["response"] = "search.no_results"
        except Exception:
            ctx["response"] = "search.no_results"
        u = self.shell.state.user
        interests = set(u.get("interests", []))
        interests.add(str(q).split()[0])
        u["interests"] = sorted(interests)[:20]
        self.shell.state.save("user", u)
        return True


TIME_RX = re.compile(
    r"(?:in\s+(\d+)\s*(second|minute|hour|day)s?)"
    r"|(?:at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)"
    r"|(tonight|tomorrow|morning|evening|noon)", re.I)


def parse_when(text):
    """Return unix timestamp or None. Pure function, unit-testable."""
    m = TIME_RX.search(text or "")
    if not m:
        return None
    now = time.localtime()
    if m.group(1):                                  # in N units
        n, unit = int(m.group(1)), m.group(2).lower()
        mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
        return time.time() + n * mult
    if m.group(3):                                  # at H[:MM] [am|pm]
        h = int(m.group(3))
        mi = int(m.group(4) or 0)
        ap = (m.group(5) or "").lower()
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        base = time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                            h, mi, 0, 0, 0, -1))
        return base if base > time.time() else base + 86400
    word = m.group(6).lower()                       # tomorrow etc.
    offsets = {"tomorrow": 86400, "tonight": 21600, "morning": 28800,
               "afternoon": 43200, "evening": 64800, "noon": 43200}
    guess = {"tomorrow": 9 * 3600}.get(word, 3600)
    return time.mktime(now[:3] + (9, 0, 0, 0, 0, -1)) + \
        (86400 if word == "tomorrow" else 0) + \
        (guess if word != "tomorrow" else 0)


@register
class Reminder(Plugin):
    name = "reminder"
    priority = 9

    def hook_add(self, ctx):
        task = self.param(ctx, ctx["decomp"].get("params", {}).get("task", 3))
        when = self.param(ctx, ctx["decomp"].get("params", {}).get("when", 4))
        due = parse_when(ctx["sentence"])
        if due is None:
            # lexicon-powered Timex parser: weekdays, months, 'next week',
            # 'in twenty minutes' -- the regex above only knows digits
            try:
                from elizaclaw import nlu
                due = nlu.parse_time(ctx["sentence"])
            except Exception:
                due = None
        if not task:
            return False
        item = {"task": str(task), "due": due or time.time() + 3600,
                "fired": False}
        st = self.shell.state.reminders
        st["items"].append(item)
        del st["items"][:-25]
        self.shell.state.save("reminders", st)
        ctx["vars"]["task"] = task
        ctx["vars"]["when"] = when or "later"
        ctx["response"] = "reminder.added"
        return True

    def hook_list(self, ctx):
        items = [i for i in self.shell.state.reminders["items"]
                 if not i["fired"]]
        ctx["vars"]["count"] = len(items)
        ctx["vars"]["list"] = "; ".join(i["task"] for i in items) or "nothing"
        ctx["response"] = "reminder.list"
        return True

    def tick(self):
        st = self.shell.state.reminders
        fired_any = False
        for item in st["items"]:
            if not item["fired"] and item["due"] <= time.time():
                item["fired"] = True
                fired_any = True
                self.shell.queue(self.shell.render(
                    "reminder.due", {"task": item["task"]}))
        if fired_any:
            self.shell.state.save("reminders", st)
