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
    """Compile an Eliza decomp pattern into a regex.

    Tokens understood:
      '*'          -> capture group (lazy unless trailing, then greedy)
      '@name'      -> synonym set from eliza.json, whole words, longest-first
      'word*'      -> prefix match (greeting patterns like 'hello*')
      literal word -> exact word match
    Anchored at the start of the sentence (classic Eliza uses COMPASS-style
    full-sentence matching), so a wildcard pattern '*' never shadows others.
    Returns (compiled_regex, num_capture_groups). Group i corresponds to
    the i-th '*' -- which is how eliza.json params like {"task": 3} are read:
    token position 3 of the original sentence, or capture order for '*'.
    """
    parts, num = [], 0
    toks = pattern.split()
    for i, tok in enumerate(toks):
        last = (i == len(toks) - 1)
        if tok == "*":
            num += 1
            parts.append("(.+)" if last else "(.+?)")
        elif tok.startswith("@"):
            syns = sorted(synonyms.get(tok[1:], []), key=len, reverse=True) \
                or [tok[1:]]
            alt = "|".join(re.escape(s) for s in syns)
            parts.append(r"(?:%s)" % alt)
        elif tok.endswith("*") and len(tok) > 1:      # prefix glob: hello*
            parts.append(r"\b%s\w*" % re.escape(tok[:-1]))
        else:
            parts.append(r"\b%s\b" % re.escape(tok))
        if not last:
            parts.append(r"\W+")
    src = "".join(parts)
    if toks and toks[-1] != "*" and not toks[-1].endswith("*"):
        src += r"\W.*|\b"                             # allow trailing words
    return re.compile(src, re.S | re.I), num


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
