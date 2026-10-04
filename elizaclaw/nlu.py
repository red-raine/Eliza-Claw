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
def compile_pattern(pattern, synonyms):
    """Compile an Eliza decomp pattern into a word-level matcher.

    Wave 0, v12 -- the final rebuild.  Post-mortem of bugs #1-#20 (logged in
    llm_wiki/decisions.md): every hand-built REGEX variant eventually failed
    on capture alignment.  Python's engine ate neighbouring atoms' spaces
    into groups; zero-width lookarounds around multi-word synonym sets were
    unmaintainable.  The real lesson from COMPASS (Weizenbaum & Bailey, 1966
    ACM session): Eliza never used regex captures at all.  It matched words,
    THEN bound segments.  So we do exactly that here with a tiny dynamic
    program over token spans.  Deterministic, dependency-free, testable.

    Grammar:
      literal word   matches one identical token
      'word*'        prefix glob ('hello*' eats 'hello', 'hellos', ...)
      '@synset'      any member of eliza.json synonyms[name]; members may be
                     multi-word and consume several tokens (longest first)
      '*'            binds ZERO OR MORE words (COMPASS semantics); multiple
                     stars split left-greedy, but always leave enough words
                     for the rest of the pattern to match
      unanchored     a pattern without leading '*' floats: it may start at
                     any token; same for trailing overflow when no '*' ends
                     the pattern (classic Eliza padded decomps with '* *').

    Returns (matcher_fn, num_stars).  matcher_fn(joined_tokens) -> list of
    captured strings (one per '*', '' allowed) or None if no match.
    """
    toks = pattern.lower().split()
    ns = sum(1 for t in toks if t == "*")

    def atom_variants(t):
        """List of token-lists this pattern atom can consume."""
        if t.startswith("@"):
            raw = list(synonyms.get(t[1:], [])) + [t[1:]]
            # bug #21: synonym members may be stored as LISTS (wordnet
            # expansion output) or plain strings -- normalise both, then
            # hash tuples to dedupe.
            norm = {tuple(str(x).split()) if isinstance(x, str)
                    else tuple(x) for x in raw}
            norm.discard(())
            return sorted(norm, key=len, reverse=True)
        if t.endswith("*") and len(t) > 1:
            return ("glob", t[:-1])                 # special-cased below
        return [t.split()]

    variants = [atom_variants(t) for t in toks]

    def consumes(a, words, i):
        """End indices j reachable by matching atom a at words[i:]."""
        kind = variants[a]
        out = []
        if isinstance(kind, tuple) and kind[0] == "glob":
            if i < len(words) and words[i].startswith(kind[1]):
                out.append(i + 1)
            return out
        for m in kind:
            L = len(m)
            if i + L <= len(words) and tuple(words[i:i + L]) == tuple(m):
                out.append(i + L)
        return sorted(set(out), reverse=True)       # longest first

    def star_ok_after(a, words, i):
        """Can pattern atoms a.. bind words starting at i (stars flexible)?"""
        memo = {}
        def go(p, q):
            if p == len(toks):
                return True
            key = (p, q)
            if key in memo:
                return memo[key]
            memo[key] = False
            if toks[p] == "*":
                for nq in range(q, len(words) + 1):
                    if go(p + 1, nq):
                        memo[key] = True
                        return True
            else:
                for e in consumes(p, words, q):
                    if go(p + 1, e):
                        memo[key] = True
                        return True
            return False
        return go(a, i)

    def match(joined):
        words = joined.split()
        nw = len(words)
        if nw == 0:
            return [""] * ns if not any(t != "*" for t in toks) else None

        # ---- recursive descent with COMPASS binding order:
        # non-star atoms take their match eagerly; a '*' takes as many words
        # as possible while still letting the SUFFIX of the pattern match
        # (leftmost-longest).  Leading float: try start positions left->right.
        result = None

        def bind(a, i, spans):
            """Match atoms[a:] against words[i:]; append star captures."""
            nonlocal result
            if result is not None:
                return True
            if a == len(toks):
                result = list(spans)
                return True
            if toks[a] == "*":
                # greedy: longest span first, but suffix must still match
                for end in range(nw, i - 1, -1):
                    if star_ok_after(a + 1, words, end):
                        # also require the exact-match path from `end`
                        if bind_exact(a + 1, end, spans + [words[i:end]]):
                            return True
                return False
            for e in consumes(a, words, i):
                if bind_exact(a + 1, e, spans):
                    return True
            return False

        def bind_exact(a, i, spans):
            """Like bind but anchored at input position i (no float)."""
            nonlocal result
            if result is not None:
                return True
            if a == len(toks):
                result = list(spans)
                return True
            if toks[a] == "*":
                for end in range(nw, i - 1, -1):
                    if star_ok_after(a + 1, words, end):
                        if bind_exact(a + 1, end, spans + [words[i:end]]):
                            return True
                return False
            for e in consumes(a, words, i):
                if bind_exact(a + 1, e, spans):
                    return True
            return False

        if toks[0] == "*":
            ok = bind_exact(0, 0, [])
        else:
            ok = False
            for st in range(nw):                    # float the start
                if bind_exact(0, st, []):
                    ok = True
                    break
        if result is None:
            return None
        caps = result
        # trailing overflow: if pattern ends with a literal and there are
        # leftover words after the last consumed atom, COMPASS patterns like
        # '* error *' already handle it via stars; plain 'i am *' captures
        # everything after.  Nothing extra needed -- bind() consumed greedily.
        #
        # bug #25 (COMPASS binding rule, Weizenbaum & Bailey 1966): when a
        # leading '*' binds zero words, the NEXT star must not swallow the
        # trigger phrase's own preposition ('remind me to | buy milk' was
        # captured as 'to buy milk').  If cap0 is empty and cap1 opens with
        # a preposition/article that the pattern itself contains, drop it.
        if ns >= 2 and caps and caps[0] == "":
            patsyn = set()
            for t in toks:
                if t.startswith("@"):
                    for m in synonyms.get(t[1:], []):
                        patsyn.update(str(m).split())
                elif t != "*":
                    patsyn.add(t)
            first = caps[1].split(" ", 1)
            if first and first[0] in patsyn \
                    and first[0] in ("to", "for", "about", "of", "on",
                                     "in", "at", "with"):
                caps[1] = first[1] if len(first) > 1 else ""
        return [" ".join(s) if isinstance(s, list) else s
                for s in caps[:ns]]

    return match, ns


def run_pattern(pattern, synonyms, joined):
    """Convenience one-shot: returns capture list or None."""
    fn, _ = compile_pattern(pattern, synonyms)
    return fn(joined)


# ------------------------------------------------------------------- intent
def _pattern_hit(pat, joined, synonyms):
    fn, _ = compile_pattern(pat, synonyms)
    return fn(joined) is not None


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
