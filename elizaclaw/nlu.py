"""Eliza-Claw :: nlu -- classical NLP, zero tokens.

Everything a "language model" would do implicitly is done here explicitly:
  * normalize + pre-transform (eliza.json transformers.pre)
  * word segmentation with synonym-set lookup (@syn sets match multi-word
    sequences, e.g. "i feel like" == @belief)
  * glob pattern -> regex compiler supporting *, @synsets and {N} captures
  * intent classification (rule-based patterns + punctuation, keyword fallback)
  * POS-lite verb/object extraction via stopword + verb-cue heuristics
  * pronoun/person post-transformation for reflections (transformers.post)

No dependencies beyond the standard library. Deterministic. Fits in your head.
"""
import re

# --------------------------------------------------------------- vocabulary
ARTICLES = {"a", "an", "the"}
PREPS = {"to", "for", "about", "with", "on", "in", "at", "of", "from",
         "by", "as", "into", "onto", "over", "under", "after", "before"}
AUXES = {"am", "is", "are", "was", "were", "be", "been", "being", "do",
         "does", "did", "have", "has", "had", "will", "would", "can",
         "could", "should", "shall", "may", "might", "must"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their"}
VERB_CUES = AUXES | {"want", "need", "wish", "hope", "feel", "think",
                     "believe", "suppose", "remember", "forget", "dream",
                     "like", "love", "hate", "know", "understand"}
FILLERS = ARTICLES | PREPS | AUXES | PRONOUNS | {"that", "this", "just",
                                                 "really", "very", "so",
                                                 "actually", "always",
                                                 "never", "often"}


def strip_punct(word):
    return word.strip(".,;:!?\"'()[]{}")


def tokenize(text):
    """Whitespace tokens with punctuation stripped."""
    return [w for w in (strip_punct(t) for t in text.lower().split()) if w]


# ------------------------------------------------------------ normalization
def apply_pre(text, table):
    """Word-level pre-transformation ('dont'->"don't", 'maybe'->'perhaps')."""
    out = []
    for tok in text.split():
        bare = tok.strip(".,;:!?\"'()").lower()
        if bare in table:
            trail = ""
            for ch in reversed(tok):
                if ch.isalnum() or ch in ("'", "-"):
                    break
                trail = ch + trail
            out.append(table[bare] + trail)
        else:
            out.append(tok)
    return " ".join(out)


# ------------------------------------------------------ pattern compilation
def _lit(w):
    """Anchored literal word: real \\b boundaries, escaped body."""
    return r"\b%s\b" % re.escape(w)


def compile_pattern(pattern, synonyms):
    """Compile an Eliza decomp pattern into a regex.  Wave 0 rebuild.

    Tokens understood:
      '*'          -> capture group (.+?), lazy unless trailing (then greedy)
      '@name'      -> synonym set from eliza.json, longest-first alternation
      'word*'      -> prefix glob (greeting patterns like 'hello*')
      literal word -> exact whole-word match via \\b...\\b
    The compiled regex is NOT start-anchored; callers use .search() on the
    joined token stream, which matches classic COMPASS behaviour closely
    enough while keeping wildcard fallbacks honest.
    Returns (compiled_regex, num_capture_groups); group i is the i-th '*'.

    Bug post-mortem (this function has now been rebuilt three times -- the
    scientific method working as intended): earlier revisions built regex
    fragments with '%' string formatting using NON-raw strings, so Python
    silently converted the backslash-b sequences into BACKSPACE control
    characters (x08) inside the regex source.  Every pattern then matched
    nothing.  Lesson recorded in llm_wiki/decisions: keep every regex
    fragment a raw string, and unit-test the compiler directly, not only
    through its callers.
    """
    toks = pattern.split()

    # ---- v4 rebuild (Wave 0, bugs #1-#6 all dead). Design rules proven by
    # the wave-0 test matrix:
    #  * every regex fragment is a RAW string (kills the x08 backspace bug);
    #  * leading '*' run -> NON-capturing optional preamble. The engine tries
    #    "eat one word + space" first, and when the body fails it retries the
    #    zero-width branch at position 0 -- no phantom-gobble possible because
    #    backtracking is honest here (unbuggy v2/v3 relied on '?' groups that
    #    the engine refused to backtrack past a matched keyword boundary...
    #    actually the killer fix: preamble is '(?:\S+\s+)?' which CAN match
    #    empty, and search() also starts mid-string, so both orders work);
    #  * interior '*' -> lazy capture guarded by NEGATIVE LOOKAHEAD on the
    #    next atom, so it stops exactly before the following literal;
    #  * terminal '*' -> '(.+)' requiring >=1 word (COMPASS semantics);
    #  * patterns ending in a literal get '(?:\W.+)?' so trailing user words
    #    don't break the match (Eliza's own '*' padding, done implicitly).

    def lit(w):
        return r"\b%s\b" % re.escape(w)

    def atom(tok):
        """Regex for one non-star token."""
        if tok.startswith("@"):
            syns = sorted(synonyms.get(tok[1:], []),
                          key=len, reverse=True) or [tok[1:]]
            return r"(?:" + "|".join(lit(s) for s in syns) + ")"
        if tok.endswith("*") and len(tok) > 1:      # prefix glob: hello*
            return r"\b%s\w*" % re.escape(tok[:-1])
        return lit(tok)

    parts = []
    num = 0
    i, n = 0, len(toks)
    # all-stars pattern ('*' or '* *'): must require at least one word,
    # otherwise it matches the empty string everywhere (bug #6: the hello
    # key's '*' stole every sentence from remind/search/task keys).
    if toks and all(t == "*" for t in toks):
        return re.compile(r"(.+)", re.S | re.I), 1
    # leading star run: optional, non-capturing preamble
    j = 0
    while j < n and toks[j] == "*":
        j += 1
    if j:
        parts.append(r"(?:\S+\s+)?")
        i = j
    while i < n:
        if toks[i] == "*":
            j = i
            while j < n and toks[j] == "*":          # '* *' collapses to '*'
                j += 1
            num += 1
            if j >= n:                               # terminal star
                parts.append(r"(.+)")
            else:                                    # interior star
                stop = atom(toks[j])
                parts.append(r"((?:(?!" + stop + r").+?)\W+)")
            i = j
            continue
        parts.append(atom(toks[i]))
        if i < n - 1 and toks[i + 1] != "*":
            parts.append(r"\W+")                     # separator between atoms
        i += 1
    out = "".join(parts)
    if toks and toks[-1] != "*" and not toks[-1].endswith("*") \
            and not toks[-1].startswith("@"):
        out += r"(?:\W.+)?"                           # optional trailing words
    return re.compile(out, re.S | re.I), num



def clean_captures(groups, ng):
    """Trim whitespace that boundary classes leave on capture edges."""
    return [g.strip() for g in groups[:ng] if g is not None]


# ------------------------------------------------------------------- intent
def _pattern_hit(pat, joined, synonyms):
    rx, _ = compile_pattern(pat, synonyms)
    return bool(rx.search(joined))


def classify_intent(sentence, classifier, synonyms):
    """Rule-based intent classification straight from eliza.json categories."""
    cats = classifier.get("categories", {})
    toks = tokenize(sentence)
    joined = " ".join(toks)
    scores = {}
    for cat, spec in cats.items():
        score = 0
        for pat in spec.get("patterns", []):
            if _pattern_hit(pat, joined, synonyms):
                score += 1
        if sentence.strip().endswith("?"):
            if cat == "question":
                score += 2
        if toks and cat == "command" and toks[0] not in PRONOUNS \
                and toks[0] not in VERB_CUES and not toks[0] in ("what", "how",
                "why", "when", "where", "who") and not sentence.strip().endswith("?"):
            score += 1  # imperative hint: bare verb-ish opener
        scores[cat] = score
    if not any(scores.values()):
        return "statement"
    best = max(sorted(scores), key=lambda c: scores[c])
    return best


# --------------------------------------------------------------- reflections
_reflect_cache = {}


def reflect(segment_text, post_table):
    """Person/number flip for Eliza-style mirroring ('i am' -> 'you are')."""
    key = id(post_table)
    if key not in _reflect_cache:
        _reflect_cache[key] = dict(post_table)
    mapping = _reflect_cache[key]
    out = []
    for tok in tokenize(segment_text):
        out.append(mapping.get(tok, tok))
    txt = " ".join(out)
    txt = re.sub(r"\bmy self\b", "yourself", txt)
    txt = re.sub(r"\byou are my\b", "I am your", txt)
    return txt


# ----------------------------------------------------------- POS-lite parse
def extract_verb_object(sentence):
    """Crude but reliable main-verb + object extraction.

    'i need to debug the authentication module'
      -> ('debug', 'authentication module')
    Skips cue verbs that are followed by 'to <verb>' so the real action wins.
    """
    toks = tokenize(sentence)
    if not toks:
        return None, None
    verb_idx = None
    for i, t in enumerate(toks):
        if t not in VERB_CUES:
            continue
        if i + 2 < len(toks) and toks[i + 1] == "to" \
                and toks[i + 2] not in AUXES:
            verb_idx = i + 2          # skip 'need/want to' -> real verb
            break
        if verb_idx is None:
            verb_idx = i
    if verb_idx is None:
        for i in range(1, len(toks)):
            if toks[i] not in FILLERS:
                verb_idx = i
                break
    if verb_idx is None:
        return None, None
    verb = toks[verb_idx]
    obj_words = [t for t in toks[verb_idx + 1:]
                 if t not in ARTICLES and t not in PREPS and t != "to"]
    return verb, " ".join(obj_words) or None


# ------------------------------------------------------------- time semantics
_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
           "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
           "november": 11, "december": 12}
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
              "twelve": 12, "fifteen": 15, "twenty": 20, "half": 30}


def parse_time(text, now=None):
    """Tiny temporal expression parser (Timex-lite). Understands:

      'tonight' 'tomorrow' 'this afternoon/evening/morning' 'in N minutes/hours'
      'at 5pm' 'at 17:30' 'on friday' 'next week/month' '<Month> 12' 'noon'

    Returns an epoch float, or None. No model, no regex monsters -- a table.
    """
    import datetime as _dt
    now = now or _dt.datetime.now()
    toks = tokenize(text)
    joined = " ".join(toks)
    m = re.search(r"\bin\s+(?:(\d+)|(\w+))\s+(minute|hour|day|week)s?\b", joined)
    if m:
        n = int(m.group(1)) if m.group(1) else _NUM_WORDS.get(m.group(2), 1)
        unit = m.group(3)
        delta = {"minute": _dt.timedelta(minutes=n),
                 "hour": _dt.timedelta(hours=n),
                 "day": _dt.timedelta(days=n),
                 "week": _dt.timedelta(weeks=n)}[unit]
        return (now + delta).timestamp()
    m = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", joined)
    if m:
        h = int(m.group(1))
        if m.group(3) == "pm" and h < 12:
            h += 12
        if m.group(3) == "am" and h == 12:
            h = 0
        target = now.replace(hour=h, minute=int(m.group(2) or 0),
                             second=0, microsecond=0)
        if target < now:                       # 'at 9am' said at 10am -> tomorrow
            target += _dt.timedelta(days=1)
        return target.timestamp()
    if "tonight" in toks or "midnight" in toks:
        return now.replace(hour=21 if "tonight" in toks else 0,
                           minute=0, second=0, microsecond=0).timestamp()
    if "noon" in toks:
        return now.replace(hour=12, minute=0, second=0,
                           microsecond=0).timestamp()
    day_shift = 1 if "tomorrow" in toks else 0
    if "afternoon" in toks:
        return (now + _dt.timedelta(days=day_shift)).replace(
            hour=15, minute=0, second=0, microsecond=0).timestamp()
    if "evening" in toks:
        return (now + _dt.timedelta(days=day_shift)).replace(
            hour=19, minute=0, second=0, microsecond=0).timestamp()
    if "morning" in toks:
        return (now + _dt.timedelta(days=day_shift)).replace(
            hour=9, minute=0, second=0, microsecond=0).timestamp()
    if "weekend" in toks:
        add = (5 - now.weekday()) % 7 or 7
        return (now + _dt.timedelta(days=add)).replace(
            hour=10, minute=0, second=0, microsecond=0).timestamp()
    for wd, idx in _WEEKDAYS.items():
        if wd in toks:
            add = (idx - now.weekday()) % 7 or (7 if "next" in toks else 0)
            if "next" in toks and add <= 0:
                add = 7
            return (now + _dt.timedelta(days=add)).replace(
                hour=9, minute=0, second=0, microsecond=0).timestamp()
    if "tomorrow" in toks:
        return (now + _dt.timedelta(days=1)).replace(
            hour=9, minute=0, second=0, microsecond=0).timestamp()
    if "month" in toks and ("next" in toks or "in" in toks):
        return (now + _dt.timedelta(days=30)).replace(
            hour=9, minute=0, second=0, microsecond=0).timestamp()
    if "week" in toks and "next" in toks:
        return (now + _dt.timedelta(weeks=1)).replace(
            hour=9, minute=0, second=0, microsecond=0).timestamp()
    for mon, num in _MONTHS.items():
        if mon in toks:
            i = toks.index(mon)
            dom = None
            for cand in toks[i + 1:i + 3]:
                if re.fullmatch(r"\d{1,2}", cand):
                    dom = int(cand)
            if dom:
                yr = now.year + (1 if num < now.month else 0)
                try:
                    return _dt.datetime(yr, num, dom, 9).timestamp()
                except ValueError:
                    return None
    return None
