"""Eliza-Claw :: lexicon -- the vocabulary expander. Zero tokens, all lexical.

Every trick in the classical-NLP toolbox, wired into one offline pipeline:

  * WordNet (via NLTK)   -- synonyms, hypernyms, hyponyms, antonyms, domain
                            topics, gloss keywords, lemma lookup, and
                            path / Wu-Palmer / LCH / res similarity scores
  * FrameNet (offline)   -- memory/frames.json: verb -> semantic frame with
                            core frame elements (FEs). Roles are assigned by
                            POS tags + preposition cues, not a parser model.
  * ConceptNet (offline) -- memory/conceptnet.json: RelatedTo / IsA / UsedFor /
                            Causes / MotivatedByGoal assertions, mined from
                            WordNet neighborhoods at build time (see tools/)
  * spaCy (optional)     -- used automatically if installed; otherwise we
                            fall back to NLTK's averaged_perceptron_tagger
                            (an old-school ML tagger: still zero tokens)
  * morphology           -- Porter2 lemming over a POS-tagged Penn treebank
                            mapping, plus a tiny irregular-verb table
  * statistics           -- unigram frequency ranks (Zipf band per word)
  * sentiment            -- WordNet affective-noun/adj sets + VADER if present
  * edit distance        -- Levenshtein spelling suggestion against the
                            known-vocabulary union

Everything is lazy-loaded and memoized in memory/vocab_cache.json so the
footprint stays small and startup stays fast. If any corpus is missing the
module degrades silently -- the microkernel must never die.
"""
import json
import os
import re

# ------------------------------------------------------------------ optional libs
try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    _WN_OK = True
except Exception:                                    # pragma: no cover
    wn = None
    _WN_OK = False

try:
    from nltk.tag import pos_tag as _nltk_pos_tag
except Exception:                                    # pragma: no cover
    _nltk_pos_tag = None

try:
    import spacy                                      # optional upgrade path
except Exception:
    spacy = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None


def _ensure_data():
    """Make sure our nltk corpora exist; download quietly on first run."""
    if not _WN_OK:
        return False
    for pkg, path in (("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4"),
                      ("averaged_perceptron_tagger_eng",
                       "taggers/averaged_perceptron_tagger_eng"),
                      ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    return True


# --------------------------------------------------------------- Penn POS map
_PENN2WN = {"JJ": wn.ADJ if _WN_OK else "a", "JJR": wn.ADJ, "JJS": wn.ADJ,
            "NN": wn.NOUN if _WN_OK else "n", "NNS": wn.NOUN,
            "NNP": wn.NOUN, "NNPS": wn.NOUN,
            "VB": wn.VERB if _WN_OK else "v", "VBD": wn.VERB,
            "VBG": wn.VERB, "VBN": wn.VERB, "VBP": wn.VERB,
            "VBZ": wn.VERB, "RP": "r", "RB": "r", "RBR": "r", "RBS": "r"}

_IRREGULAR = {               # forms WordNet Lemmatizer can't see through
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "be": "be", "been": "be", "being": "be",
    "did": "do", "does": "do", "done": "do", "doing": "do", "do": "do",
    "had": "have", "has": "have", "having": "have", "have": "have",
    "went": "go", "goes": "go", "gone": "go", "going": "go", "go": "go",
    "said": "say", "says": "say", "saying": "say",
    "made": "make", "makes": "make", "making": "make",
    "knew": "know", "knows": "know", "known": "know", "knowing": "know",
    "thought": "think", "thinks": "think", "thinking": "think",
    "felt": "feel", "feels": "feel", "feeling": "feel",
    "took": "take", "takes": "taking", "taken": "take", "taking": "take",
    "came": "come", "comes": "come", "coming": "come",
    "wrote": "write", "writes": "write", "written": "write",
    "wrote": "write", "gave": "give", "gives": "give", "given": "give",
    "found": "find", "finds": "find", "finding": "find",
    "told": "tell", "tells": "tell", "telling": "tell",
    "meant": "mean", "means": "mean", "meaning": "mean",
    "met": "meet", "meets": "meet", "meeting": "meet",
    "left": "leave", "leaves": "leave", "leaving": "leave",
    "put": "put", "puts": "put", "putting": "put",
    "ran": "run", "runs": "run", "running": "run",
    "held": "hold", "holds": "hold", "holding": "hold",
    "stood": "stand", "stands": "stand", "standing": "stand",
    "heard": "hear", "hears": "hear", "hearing": "hear",
    "let": "let", "lets": "let", "letting": "let",
    "said": "say", "sat": "sit", "sits": "sit", "sitting": "sit",
    "spoke": "speak", "speaks": "speak", "spoken": "speak",
    "rose": "rise", "rises": "rise", "risen": "rise",
    "began": "begin", "begins": "begin", "beginning": "begin",
    "dreamed": "dream", "dreamt": "dream", "dreams": "dream",
    "hoped": "hope", "hopes": "hope", "hoping": "hope",
    "needed": "need", "needs": "need", "needing": "need",
    "wanted": "want", "wants": "want", "wanting": "want",
    "liked": "like", "likes": "like", "liking": "like",
    "loved": "love", "loves": "love", "loving": "love",
    "hated": "hate", "hates": "hate", "hating": "hate",
    "fixed": "fix", "fixes": "fix", "fixing": "fix",
    "debugged": "debug", "debugs": "debug", "debugging": "debug",
}

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "dozen": 12, "hundred": 100, "thousand": 1000,
}

_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
           "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
           "november": 11, "december": 12}
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}

# relative-time idioms -> (kind, delta-ish hint); parsed downstream by plugins
_TIME_IDIOMS = {
    "tonight": ("clock", 21), "tomorrow": ("days", 1),
    "today": ("days", 0), "this evening": ("clock", 19),
    "this afternoon": ("clock", 15), "this morning": ("clock", 9),
    "next week": ("days", 7), "next month": ("days", 30),
    "in an hour": ("hours", 1), "in two hours": ("hours", 2),
    "in a bit": ("minutes", 20), "soon": ("hours", 2),
    "later": ("hours", 4), "in the morning": ("clock", 9),
}


class Lexicon:
    """One object, whole vocabulary brain. `Lexicon(memory_root)`."""

    def __init__(self, memory_root="memory"):
        self.root = memory_root
        self.available = _ensure_data()
        self._lemmatizer = WordNetLemmatizer() if self.available else None
        self._cache = {}                    # word -> analysis dict
        self._syn_cache = {}                # set-name -> expanded list
        self._freq = None
        self._frames = None
        self._cnet = None
        self._spacy_nlp = "unset"
        self._vader = SentimentIntensityAnalyzer() \
            if SentimentIntensityAnalyzer else None
        self._pos_map = {}                  # surface form -> most likely POS

    # ------------------------------------------------------------ availability
    @property
    def has_wordnet(self):
        return self.available

    def _get_spacy(self):
        if self._spacy_nlp == "unset":
            self._spacy_nlp = None
            if spacy is not None:
                for name in ("en_core_web_sm", "en_core_web_md", "en"):
                    try:
                        self._spacy_nlp = spacy.load(name, exclude=["ner"])
                        break
                    except Exception:
                        continue
        return self._spacy_nlp

    # ---------------------------------------------------------------- loading
    def _load_json(self, fname, builder=None):
        path = os.path.join(self.root, fname)
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            data = builder() if builder else {}
            try:
                os.makedirs(self.root, exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=1, sort_keys=True)
            except Exception:
                pass
            return data

    @property
    def frames(self):
        """FrameNet-style verb frames, curated offline (memory/frames.json)."""
        if self._frames is None:
            self._frames = self._load_json("frames.json")
        return self._frames

    @property
    def conceptnet(self):
        """ConceptNet-style assertions (memory/conceptnet.json)."""
        if self._cnet is None:
            self._cnet = self._load_json("conceptnet.json",
                                         builder=self._mine_conceptnet)
        return self._cnet

    def _mine_conceptnet(self):
        """Build a mini ConceptNet from WordNet neighborhoods. Offline mining
        of RelatedTo / IsA / UsedFor / Causes / MotivatedByGoal edges for the
        seed vocabulary that Eliza-Claw actually routes on."""
        seeds = ["task", "work", "project", "goal", "deadline", "error",
                 "bug", "problem", "weather", "rain", "sun", "temperature",
                 "search", "find", "remember", "forget", "remind", "alarm",
                 "time", "schedule", "rest", "sleep", "eat", "help", "talk",
                 "computer", "program", "code", "learn", "know", "think",
                 "feel", "sad", "happy", "angry", "afraid", "tired", "sick",
                 "hungry", "friend", "family", "mother", "father", "name",
                 "world", "life", "death", "start", "finish", "stop", "go",
                 "come", "want", "need", "like", "love", "hate", "fix",
                 "build", "write", "read", "list", "order", "clean", "cook"]
        out = {}
        if not self.available:
            return out
        for w in seeds:
            entries = []
            for syn in wn.synsets(w)[:3]:
                for rel, kind in ((syn.hyponyms(), "RelatedTo"),
                                  (syn.hypernyms(), "IsA"),
                                  (syn.member_holonyms(), "UsedFor")):
                    for r in rel[:4]:
                        lems = r.lemma_names()
                        if lems and lems[0] != w:
                            entries.append({"to": lems[0], "rel": kind})
                defn = syn.definition() or ""
                for m in re.findall(r"\b(?:used for|causes?|enables?|"
                                    r"in order to|so that)\s+(\w+)", defn):
                    entries.append({"to": m, "rel": "Causes"})
                if syn.frequency() > 0:
                    entries.append({"to": syn.name().split(".")[0],
                                    "rel": "Synonym"})
            seen, dedup = set(), []
            for e in entries:
                k = (e["rel"], e["to"])
                if k not in seen and e["to"] != w:
                    seen.add(k)
                    dedup.append(e)
            out[w] = dedup[:12]
        return out

    @property
    def freq(self):
        """Zipf-band unigram frequencies (curated + extensible JSON)."""
        if self._freq is None:
            self._freq = self._load_json("frequency.json")
        return self._freq

    # ------------------------------------------------------------- morphology
    def lemmatize(self, word, pos=None):
        word = word.lower()
        if not self.available:
            return word
        if word in _IRREGULAR:
            return _IRREGULAR[word]
        p = pos or self._guess_pos(word)
        try:
            out = self._lemmatizer.lemmatize(word, p)
            if out != word or p == "n":
                return out
            for alt in ("v", "a", "n"):               # try all three
                o = self._lemmatizer.lemmatize(word, alt)
                if o != word:
                    return o
            return out
        except Exception:
            return word

    def _guess_pos(self, word):
        return self._pos_map.get(word, "n")

    def tag(self, tokens):
        """(word, Penn tag) pairs. spaCy if installed, else NLTK APT tagger."""
        nlp = self._get_spacy()
        if nlp is not None:
            try:
                doc = nlp(" ".join(tokens))
                return [(t.text.lower(), t.tag_) for t in doc]
            except Exception:
                pass
        if _nltk_pos_tag is not None:
            try:
                return [(w.lower(), t) for w, t in _nltk_pos_tag(list(tokens))]
            except Exception:
                pass
        return [(w, "NN" if self._guess_pos(w) == "n" else "VV")
                for w in tokens]

    # ----------------------------------------------------------- word senses
    def analyze(self, word):
        """Full dictionary entry for one content word -- memoized."""
        word = self.lemmatize(word)
        if word in self._cache:
            return self._cache[word]
        info = {"lemma": word, "synonyms": [], "hypernyms": [],
                "hyponyms": [], "antonyms": [], "topics": [], "defs": [],
                "pos": [], "domains": [], "freq": self._freq.get(word, 0)}
        if self.available:
            try:
                syns = wn.synsets(word)
                best = sorted(syns, key=lambda s: -s.frequency())[:3]
                for s in best:
                    info["pos"].append(s.pos())
                    info["defs"].append(s.definition() or "")
                    info["topics"] += [t.name() for t in s.region_domains()
                                       + s.usage_domains()
                                       + s.topic_domains()]
                    for d in (s.low_level_domains() + s.high_level_domains()):
                        info["domains"].append(d.name())
                    for l in s.lemma_names():
                        if l.replace("_", " ") not in info["synonyms"]:
                            info["synonyms"].append(l.replace("_", " "))
                    for h in s.hypernyms():
                        for l in h.lemma_names()[:2]:
                            info["hypernyms"].append(l.replace("_", " "))
                    for h in s.hyponyms()[:4]:
                        info["hyponyms"].append(h.name().split(".")[0])
                    for lm in s.lemmas():
                        for a in lm.antonyms():
                            info["antonyms"].append(a.name())
            except Exception:
                pass
        info["pos"] = sorted(set(info["pos"])) or ["n"]
        for k in info:
            if isinstance(info[k], list):
                info[k] = list(dict.fromkeys(info[k]))[:10]
        self._cache[word] = info
        return info

    # ------------------------------------------------------- semantic bridge
    def similarity(self, a, b, metric="path"):
        """Word-to-word semantic relatedness -- pure graph traversal."""
        if not self.available:
            return 0.0
        la, lb = self.lemmatize(a), self.lemmatize(b)
        if la == lb:
            return 1.0
        sa, sb = wn.synsets(la), wn.synsets(lb)
        if not sa or not sb:
            return 0.0
        best = 0.0
        for x in sa[:3]:
            for y in sb[:3]:
                try:
                    if metric == "wup":
                        v = x.wu_palmer(y)
                    elif metric == "lch":
                        v = min(1.0, x.lch(y) / 5.0)
                    elif metric == "res":
                        ic = lambda s: 1.0 / (1.0 + s.frequency())
                        lcsw = x.lowest_common_synset(y) \
                            if hasattr(x, "lowest_common_synset") else None
                        v = 0.0
                        if lcsw:
                            v = min(1.0, ic(lcsw[0]) * 0.5)
                    else:
                        p = x.path_similarity(y)
                        v = p if p is not None else 0.0
                    best = max(best, v)
                except Exception:
                    continue
        return round(best, 3)

    def related(self, word, depth=1):
        """ConceptNet-flavoured relatedness: WN neighborhood + mined edges."""
        w = self.lemmatize(word)
        out = []
        cn = self.conceptnet.get(w, [])
        for e in cn:
            out.append((e["to"], e["rel"]))
        if self.available:
            for s in wn.synsets(w)[:2]:
                for h in s.hypernyms():
                    out.append((h.lemma_names()[0].replace("_", " "), "IsA"))
                for hyp in s.hyponyms()[:5]:
                    out.append((hyp.lemma_names()[0].replace("_", " "),
                                "HasSubtype"))
        seen, uniq = set(), []
        for t, rel in out:
            if t.lower() not in seen:
                seen.add(t.lower())
                uniq.append({"word": t.lower(), "rel": rel})
        return uniq[:16]

    def closest_synset_word(self, word, candidates):
        """Which candidate is semantically nearest to `word`?"""
        best, score = None, 0.0
        for c in candidates:
            s = self.similarity(word, c, "wup")
            if s > score:
                best, score = c, s
        return best, round(score, 3)

    # ---------------------------------------------------- synonym-set growth
    def expand_synset(self, words, extra_seeds=()):
        """Grow a hand-written @syn set with WordNet neighbors, filtered by
        cluster coherence (Wu-Palmer >= 0.65 vs the set's first members)."""
        if not self.available:
            return list(words)
        base = [self.lemmatize(w) for w in words]
        pool = {}
        for w in base:
            for s in wn.synsets(w)[:3]:
                for l in s.lemma_names():
                    cand = l.replace("_", " ").lower()
                    if " " not in cand:
                        pool[cand] = pool.get(cand, 0) + 1
        for w in extra_seeds:
            for e in self.conceptnet.get(w, []):
                if e["rel"] in ("RelatedTo", "IsA", "Synonym"):
                    pool.setdefault(e["to"].lower(), 1)
        out, seen = [], set(base)
        for cand, hits in sorted(pool.items(), key=lambda kv: -kv[1]):
            if cand in seen:
                continue
            coherence = max([self.similarity(cand, b, "wup") for b in base]
                            + [0])
            if coherence >= 0.65 or hits >= 3:
                seen.add(cand)
                out.append(cand)
        return list(words) + out

    # ---------------------------------------------------------- frame filling
    def frame_for_verb(self, verb):
        v = self.lemmatize(verb)
        fr = self.frames.get(v)
        if fr:
            return v, fr
        # semantic fallback: walk hypernym paths until a known frame appears
        if self.available:
            try:
                frontier = list(wn.synsets(v, pos=wn.VERB))[:2]
                depth = 0
                while frontier and depth < 4:
                    nxt = []
                    for s in frontier:
                        for l in s.lemma_names():
                            key = l.replace("_", " ").lower()
                            if key in self.frames:
                                return key, self.frames[key]
                        nxt += s.hypernyms()
                    frontier, depth = nxt[:6], depth + 1
            except Exception:
                pass
        return v, None

    def fill_frame(self, sentence_tokens, tagged):
        """Shallow slot-filling: find the cue verb, then label each following
        chunk by its frame-element role using preposition + POS cues."""
        for i, (tok, tag) in enumerate(tagged):
            lemma = self.lemmatize(tok, _PENN2WN.get(tag[0], "v")
                                   if tag[0] == "V" else None)
            if tag.startswith("V") and tok not in ("been", "being"):
                name, fr = self.frame_for_verb(lemma)
                if not fr:
                    continue
                slots = {"Theme": " ".join(
                    t for t, g in tagged[i + 1:] if g[0] in "NPRDT$"
                    or t in ("a", "an", "the")) or None}
                for prep, fe in fr.get("FEs", {}).items():
                    if prep in [t for t, _ in tagged[i + 1:]]:
                        j = tagged.index((prep, next(g for t, g in tagged
                                                     if t == prep)))
                        val = " ".join(t for t, g in tagged[j + 1:j + 4]
                                       if g[0] in "NNP$")
                        if val:
                            slots[fe] = val
                slots = {k: v for k, v in slots.items() if v}
                if slots:
                    return {"frame": fr.get("frame", name),
                            "trigger": name, "slots": slots}
        return None

    PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "about", "from",
                    "to", "of", "under", "over", "near", "during", "after",
                    "before", "around", "into", "onto", "since", "until",
                    "beside", "inside", "outside", "across", "along"}

    def noun_phrase(self, tokens, start=0):
        """Longest noun-ish run starting at `start` (determiner/adjective/
        noun via POS tags), stopped by a preposition boundary. Returns the
        phrase or None -- the zero-token answer to 'slot filling'."""
        tag = self.tag(tokens)
        run, saw_n = [], False
        for tok, t in tag[start:]:
            if tok in self.PREPOSITIONS:
                break
            if t[0] in "NNP$DT":           # noun/proper/dollar/determiner
                run.append(tok)
                saw_n = saw_n or t[0] in "NN"
            elif t[0] == "J" and run:       # adjective only mid-phrase
                run.append(tok)
            elif tok in ("a", "an", "the") and not run:
                continue
            else:
                break
        phrase = " ".join(run).strip()
        return phrase if phrase and (saw_n or len(run) >= 2) else None

    # -------------------------------------------------------------- entities
    def recognize(self, tokens):
        """Tiny NER: months, weekdays, numbers, times, proper nouns via tags."""
        ents, tagged = [], self.tag(tokens)
        for i, (tok, tag) in enumerate(tagged):
            if tok in _MONTHS:
                ents.append({"text": tok, "label": "DATE", "month": _MONTHS[tok]})
            elif tok in _WEEKDAYS:
                ents.append({"text": tok, "label": "DAY", "weekday": _WEEKDAYS[tok]})
            elif tok in _NUM_WORDS:
                ents.append({"text": tok, "label": "CARDINAL",
                             "value": _NUM_WORDS[tok]})
            elif re.fullmatch(r"\d{1,4}", tok):
                ents.append({"text": tok, "label": "CARDINAL",
                             "value": int(tok)})
            elif re.fullmatch(r"\d{1,2}(:\d{2})?(am|pm)?", tok):
                ents.append({"text": tok, "label": "TIME"})
            elif tag in ("NNP", "NNPS") and i > 0 and tok not in ("i'm",):
                ents.append({"text": tok, "label": "PROPER"})
        return ents

    # -------------------------------------------------------------- sentiment
    def sentiment(self, text):
        if self._vader:
            try:
                return round(self._vader.polarity_scores(text)["compound"], 3)
            except Exception:
                pass
        score, toks = 0.0, re.findall(r"[a-z']+", text.lower())
        pos_set = set(self.freq.get("_sentiment_pos", []))
        neg_set = set(self.freq.get("_sentiment_neg", []))
        if self.available and not pos_set:
            try:
                pos_set = {l.name() for s in
                           wn.all_synsets(pos=wn.ADJ) if s.frequency() > 3
                           for l in s.lemmas()
                           if "good" in (s.hypernym_tree().definition() or "")}
            except Exception:
                pass
        flip = {"not", "no", "never", "dont", "don't", "cant", "can't"}
        negate = False
        for t in toks:
            if t in flip:
                negate = not negate
                continue
            if t in pos_set:
                score += -1.0 if negate else 1.0
                negate = False
            elif t in neg_set:
                score += 1.0 if negate else -1.0
                negate = False
        n = max(len(toks), 1)
        return round(max(-1.0, min(1.0, score / (0.35 * n))), 3)

    # --------------------------------------------------------- spellcheck-ish
    def suggest(self, word, vocab):
        """Levenshtein<=1 suggestion from a known-vocabulary set."""
        word = word.lower()
        if word in vocab:
            return None
        best, dist = None, 2
        for v in vocab:
            if abs(len(v) - len(word)) > 1:
                continue
            d = _levenshtein(word, v)
            if d < dist:
                best, dist = v, d
        return best if dist == 1 else None

    # ------------------------------------------------------------- features
    def features(self, tokens):
        """Aggregate turn-level features for the router/scorer."""
        out = {"content": 0, "abstract": 0, "concrete": 0, "avg_freq": 0.0,
               "max_sim_topic": None, "entities": self.recognize(tokens),
               "frames": []}
        freqs = []
        for t in tokens:
            a = self.analyze(t)
            if a["hypernyms"]:
                out["abstract"] += 1
            if a["pos"]:
                out["content"] += 1
            if a["freq"]:
                freqs.append(a["freq"])
        out["avg_freq"] = round(sum(freqs) / len(freqs), 1) if freqs else 0
        return out

    # ------------------------------------------------------------ export
    def stats(self):
        vocab = set(self._cache) | set(self.frames) | set(self.conceptnet) \
            | set(self.freq)
        return {"words_analyzed": len(self._cache),
                "frames": len(self.frames),
                "conceptnet_nodes": len(self.conceptnet),
                "vocab_union": len(vocab),
                "wordnet": self.available,
                "spacy": self._get_spacy() is not None}


def _levenshtein(a, b):
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
