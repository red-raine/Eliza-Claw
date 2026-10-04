"""Eliza-Claw :: clan -- CLAN-flavored scripting for a zero-token microkernel.

Historical mining (docs/HISTORY.md, Wave 3): the MUD era gave us two gifts.
  * MDI's CLAN language (Timothy John Martin, late 80s/90s): pattern-action
    rules with %variables and wildcard captures -- literally Eliza's idea
    promoted to a programming language.
  * MOO's verb system: every object owns verbs, `pass` falls through to a
    parent, `$foo` names a global, and programmers typed `program` to edit
    live. Warm mud bots like LambdaMoo's agents lived here.

So .clan files are our love child of both, plus ELIZA's decomp table, plus
Wumpus-world rule syntax, plus SHRINKU-style list comprehension:

    # hello.clan -- place in scripts/
    @greet = hello hi hey yo sup howdy
    ^@greet*            => "Hey there, {name}!"          # template
    ^my (.* )is (.*)    => "Tell me more about $2."      # $1..$n captures
    ^remind me to (.*) in (\d+) minutes
                        => +reminder.minutes=$2 subject=$1
                        => "Done. I'll ping you in $2 minutes."
    ^who are you        => !shell.plugins["identity"].hook_whoami
    ^\d+ \+ \d+         => =eval($0)  => "$0 is {result}"

Grammar (one directive per line, '#' comments):
    @name = word word word     synonym set (usable as @name in patterns)
    ^pattern [=> action]*      anchored pattern; '*' wildcards capture $1..
                               bare trailing '=>' string is the reply
    action forms:
        key=value[|key=value]  call plugin hook: name.hook_<key>(...)
        !dotted.path           run a shell action (same as eliza.json agents)
        =expr                  safe arithmetic eval (python eval, no builtins)
        "template"             emit mad-lib text ($n + {var} substitution)
    priority = N               route priority for rules defined below
    topic = name               default response topic prefix

The compiler emits exactly the dict shape eliza.json "keys" uses, so the
router doesn't care whether a rule was hand-written in JSON or scripted in
CLAN. That is the whole trick: one kernel, many authoring languages.
"""
import os
import re


class ClanError(Exception):
    pass


_VAR = re.compile(r"\{(\w+)\}")


def _to_decomp(pattern):
    """CLAN regex-ish pattern -> Eliza decomp tokens.
    '(.*)' groups become '*'; literal words stay literal."""
    p = pattern.strip()
    if p.startswith("^"):
        p = p[1:]
    chunks = [c.strip() for c in re.split(r"\(\.\*\)|\(\.\+\)", p)]
    rebuilt = []
    for ci, ch in enumerate(chunks):
        for w in ch.split():
            if w.startswith("\\"):
                continue
            rebuilt.append(w.replace("$", ""))
        if ci < len(chunks) - 1:
            rebuilt.append("*")
    return " ".join(rebuilt)


def compile_clan(text, source="<clan>"):
    """Compile .clan script text into an eliza.json-compatible key spec."""
    keys, synonyms, notes = [], {}, []
    cur_priority = 0
    cur_topic = None
    lines = text.splitlines()
    pending = None
    for ln, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        s = line.strip()
        if s.startswith("@"):                       # synonym set
            m = re.match(r"@(\w+)\s*=\s*(.+)", s)
            if not m:
                raise ClanError("%s:%d bad @set" % (source, ln))
            synonyms[m.group(1)] = m.group(2).split()
            continue
        if s.startswith("priority"):
            cur_priority = int(s.split("=")[1])
            continue
        if s.startswith("topic"):
            cur_topic = s.split("=")[1].strip()
            continue
        if s.startswith("^"):                       # new rule
            pat = s[1:]
            segs = [x.strip() for x in pat.split("=>")]
            pattern_raw, actions = segs[0], segs[1:]
            pending = {"pattern": _to_decomp(pattern_raw),
                       "_raw": pattern_raw, "actions": [],
                       "line": ln, "source": source}
            for a in actions:
                pending["actions"].append(_parse_action(a, source, ln))
            continue
        if s.startswith("=>") and pending is not None:
            pending["actions"].append(_parse_action(s[2:], source, ln))
            continue
        if pending is not None:                     # finalize on next rule
            keys.append(_finalize(pending, synonyms, cur_priority, cur_topic))
            pending = None
            # fallthrough: treat this line as a fresh directive
            raise ClanError("%s:%d unexpected line: %r" % (source, ln, s))
    if pending is not None:
        keys.append(_finalize(pending, synonyms, cur_priority, cur_topic))
    return {"keys": keys, "synonyms": synonyms, "notes": notes}


def _parse_action(s, source, ln):
    s = s.strip()
    if s.startswith('"') or s.startswith("'"):
        return {"kind": "reply", "template": s.strip("\"'")}
    if s.startswith("!"):
        return {"kind": "shell", "path": s[1:].strip()}
    if s.startswith("="):
        expr = s[1:].strip()
        if not re.fullmatch(r"[\d\s\.\+\-\*/%\(\)]+", expr):
            raise ClanError("%s:%d unsafe expression: %r" % (source, ln, expr))
        return {"kind": "calc", "expr": expr}
    m = re.match(r"(\w+)=(\S+)", s)
    if m:                                            # hook arg form k=v
        return {"kind": "hook", "args": _parse_kv(s)}
    raise ClanError("%s:%d cannot parse action %r" % (source, ln, s))


def _parse_kv(s):
    out = {}
    for pair in s.split("|"):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _finalize(p, synonyms, priority, topic):
    key = {
        "keyword": "xnone",
        "decomps": [{"pattern": p["pattern"]}],
        "priority": priority,
        "plugin": "clan",
        "clan": {"source": p["source"], "actions": p["actions"],
                 "topic": topic},
    }
    return key


def load_scripts(shell, folder="scripts"):
    """Import every *.clan in folder, merge compiled keys into shell.cfg.
    Returns list of sources loaded. Live-reloadable: call again any time."""
    loaded = []
    if not os.path.isdir(folder):
        return loaded
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".clan"):
            continue
        path = os.path.join(folder, fn)
        try:
            with open(path, encoding="utf-8") as fh:
                comp = compile_clan(fh.read(), source=fn)
        except ClanError as e:
            shell.queue("[clan] %s: compile error: %s" % (fn, e))
            continue
        shell.cfg["synonyms"].update(comp["synonyms"])
        shell.cfg["keys"].extend(comp["keys"])
        loaded.append(fn)
    return loaded
