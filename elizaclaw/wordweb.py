"""Eliza-Claw :: wordweb -- the Bayesian ternary word-net router.

What came out of the history-mining (see docs/HISTORY.md):

  * 1956: Selfridge's Pandemonium -> specialists vote, scores propagate.
  * 1960s: Minsky frames + Quillian's semantic memory (spreading activation).
  * 1970s: Colby's PARRY believes things; Weizenbaum's Eliza matches patterns.
  * 1982: Werbos invents backpropagation; we do it in numpy, by hand, on a
           3-layer net so small you could draw it on a napkin.
  * 1988: Pearl popularizes Bayesian networks; our priors are literal beliefs.
  * 1990s: Naive Bayes spam filters (Sahami et al.) -> Laplace-smoothed counts.

The network is a tiny ternary-weight MLP over the WordWeb vocabulary:

    input   = bag-of-words presence vector (ternary-ish features)
    hidden = ReLU layer (numpy matmul)          <- perceptron 1958, backprop 1982
    output  = softmax over ROUTE classes         <- decision theory 1954

Classes (the "routes" a turn can take):
    task      create/delete/list/done work items
    reminder  timed promises (remind me at 5pm ...)
    info      weather / search / time facts
    chat      Eliza-style reflective conversation
    system    help, quit, memory, config, status

Training data lives in data/routes.jsonl (one {"text","label"} per line).
Rebuild the model any time with:  python -m elizaclaw.tools.build_wordweb

Everything degrades gracefully: no numpy -> hash-score fallback; no WordNet
-> unigram statistics only. The microkernel must never die.
"""
import json
import math
import os
import re

try:
    import numpy as np
except Exception:                                    # pragma: no cover
    np = None

ROUTES = ["task", "reminder", "info", "chat", "system"]
_STOPWORDS = set("""a an the and or but if while is are was were be been being
do does did doing have has had having i you he she it we they me him her them
my your his its our their this that these those of in on at to for with from
by as about into over under just very so than then there here what when how
why who whom which whose will would can could may might must shall should
not no nor yes ok okay please thanks thank am""".split())

_TOKEN_RX = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return _TOKEN_RX.findall(text.lower())


def content_words(text):
    return [w for w in tokenize(text) if w not in _STOPWORDS]


# --------------------------------------------------------------- vocab build
def build_vocab(corpus, lex=None, min_df=1):
    """WordWeb vocabulary: every corpus word + lemmas + WordNet synonyms
    + hypernym roots + slang aliases. Compact: one list, one dict."""
    df = {}
    for ex in corpus:
        for w in set(content_words(ex["text"])):
            df[w] = df.get(w, 0) + 1
    vocab = {w for w, c in df.items() if c >= min_df}
    if lex is not None:
        grown = set()
        for w in list(vocab):
            try:
                grown.add(lex.lemmatize(w))
                for e in lex.conceptnet.get(w, [])[:4]:
                    grown.add(e["to"])
            except Exception:
                pass
        vocab |= {w for w in grown if w not in _STOPWORDS}
    words = sorted(vocab)
    return {w: i for i, w in enumerate(words)}, words


class WordWeb:
    """Ternary-weighted semantic web + Bayesian route prior + tiny MLP."""

    def __init__(self, root="data", lex=None):
        self.root = root
        self.lex = lex
        self.vocab = {}
        self.words = []
        self.w3 = None            # (n_terms, n_routes) ternary evidence matrix
        self.prior = None         # Bayesian class prior
        self.counts = None        # raw counts for online updates
        self.total = 0
        self.net = None           # trained MLP weights {W1,b1,W2,b2}
        self.meta = {}
        self._dirty = False

    # ---------------------------------------------------------------- io
    def _p(self, name):
        return os.path.join(self.root, name)

    def load(self):
        try:
            with open(self._p("wordweb.json"), encoding="utf-8") as fh:
                d = json.load(fh)
            self.vocab = d["vocab"]
            self.words = d["words"]
            self.w3 = [[float(x) for x in row] for row in d["w3"]] \
                if d.get("w3") else None
            self.prior = d["prior"]
            self.counts = d["counts"]
            self.total = d["total"]
            self.net = d.get("net")
            self.meta = d.get("meta", {})
            return True
        except Exception:
            return False

    def save(self):
        os.makedirs(self.root, exist_ok=True)
        d = {"vocab": self.vocab, "words": self.words,
             "w3": [[int(round(x)) for x in row] for row in self.w3]
             if self.w3 else None,
             "prior": self.prior, "counts": self.counts,
             "total": self.total, "net": self.net, "meta": self.meta}
        with open(self._p("wordweb.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh, separators=(",", ":"))
        self._dirty = False

    # ------------------------------------------------------------ training
    def fit_stats(self, corpus):
        """Naive-Bayes counts + ternary evidence matrix over the vocabulary."""
        self.counts = [[0.0] * len(ROUTES) for _ in self.words]
        self.prior = [0.0] * len(ROUTES)
        for ex in corpus:
            y = ROUTES.index(ex["label"])
            self.prior[y] += 1
            for w in set(content_words(ex["text"])):
                if w in self.vocab:
                    self.counts[self.vocab[w]][y] += 1
        self.total = sum(self.prior)
        self.prior = [(c + 1.0) / (self.total + len(ROUTES))
                      for c in self.prior]
        self._rebuild_w3()

    def _rebuild_w3(self):
        """Ternarize each term-route posterior log-odds into {-1,0,+1}xfloat.
        Sign = evidence for/against the route; magnitude kept as confidence."""
        nr = len(ROUTES)
        self.w3 = []
        for row in self.counts:
            tot = sum(row) + nr
            new = []
            for j, c in enumerate(row):
                post = (c + 0.5) / tot
                odds = math.log(post / (1.0 - post))       # vs complement
                conf = min(abs(odds) / 2.0, 1.0)
                new.append(conf if odds > 0.15 else
                           (-conf if odds < -0.15 else 0.0))
            self.w3.append(new)

    def train_mlp(self, corpus, hidden=24, epochs=200, lr=0.15, seed=7):
        """Hand-written backprop (Werbos 1982 style) on a 3-layer net.
        Ternary inputs: presence=+1, stopword-absent=0. One-hot targets."""
        if np is None:
            return False
        rng = np.random.default_rng(seed)
        n, nh, nc = len(self.words), hidden, len(ROUTES)
        scale = 1.0 / math.sqrt(n)
        W1 = rng.normal(0, scale, (n, nh))
        b1 = np.zeros(nh)
        W2 = rng.normal(0, 1.0 / math.sqrt(nh), (nh, nc))
        b2 = np.zeros(nc)
        X = np.zeros((len(corpus), n))
        Y = np.zeros((len(corpus), nc))
        for i, ex in enumerate(corpus):
            for w in set(content_words(ex["text"])):
                if w in self.vocab:
                    X[i, self.vocab[w]] = 1.0
            Y[i, ROUTES.index(ex["label"])] = 1.0
        m = len(corpus)
        for ep in range(epochs):
            z1 = X @ W1 + b1
            h = np.maximum(z1, 0.0)                        # ReLU
            logits = h @ W2 + b2
            e = np.exp(logits - logits.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
            dlog = (p - Y) / m
            dW2 = h.T @ dlog + 1e-4 * W2
            db2 = dlog.sum(0)
            dh = dlog @ W2.T
            dh[z1 <= 0] = 0.0
            dW1 = X.T @ dh + 1e-4 * W1
            db1 = dh.sum(0)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        loss = -np.log(p[np.arange(m), Y.argmax(1)] + 1e-9).mean()
        acc = float((p.argmax(1) == Y.argmax(1)).mean())
        self.net = {"W1": W1.tolist(), "b1": b1.tolist(),
                    "W2": W2.tolist(), "b2": b2.tolist(),
                    "hidden": nh}
        self.meta.update({"mlp_train_acc": round(acc, 4),
                          "mlp_train_loss": round(float(loss), 4),
                          "vocab_size": n, "corpus_size": m})
        return True

    # ------------------------------------------------------------- inference
    def _vec(self, tokens):
        v = [0.0] * len(self.words)
        for w in tokens:
            i = self.vocab.get(w)
            if i is not None:
                v[i] = 1.0
        return v

    def bayes_scores(self, tokens):
        """Posterior per route from the ternary evidence matrix."""
        idx = [self.vocab[w] for w in tokens if w in self.vocab]
        scores = list(self.prior)
        for j in range(len(ROUTES)):
            s = math.log(scores[j])
            for i in idx:
                col = self.w3[i][j]
                s += 0.6 * col               # likelihood tempering
            scores[j] = s
        mx = max(scores)
        ex = [math.exp(s - mx) for s in scores]
        tot = sum(ex) or 1.0
        return {r: round(e / tot, 4) for r, e in zip(ROUTES, ex)}

    def mlp_scores(self, tokens):
        if not self.net or np is None:
            return None
        x = np.array(self._vec(tokens))
        W1 = np.array(self.net["W1"]); b1 = np.array(self.net["b1"])
        W2 = np.array(self.net["W2"]); b2 = np.array(self.net["b2"])
        h = np.maximum(W1.T.dot(x) + b1, 0.0)
        z = W2.T.dot(h) + b2
        e = np.exp(z - z.max())
        p = e / e.sum()
        return {r: round(float(v), 4) for r, v in zip(ROUTES, p)}

    def route(self, text, blend=(0.35, 0.45, 0.20)):
        """Ensemble: Bayesian word-web + backprop MLP + keyword programmatic.
        Returns (route, confidence, full score table)."""
        toks = content_words(text)
        bay = self.bayes_scores(toks) if self.w3 else \
            {r: 1.0 / len(ROUTES) for r in ROUTES}
        mlp = self.mlp_scores(toks)
        kw = self.knowledgeable_scores(text)
        parts, ws = [bay, kw], [blend[0], blend[2]]
        if mlp:
            parts.append(mlp); ws.insert(1, blend[1])
        tot_w = sum(ws)
        comb = {r: sum(w * p[r] for w, p in zip(ws, parts)) / tot_w
                for r in ROUTES}
        # Pandemonium stage (Selfridge 1956): loud simple demons shout over
        # the statistical chorus. A strong exact-phrase hit is a demand,
        # not a suggestion -- it wins outright above its threshold.
        LOUD = {"task": 8, "reminder": 8, "info": 8, "system": 7, "chat": 6}
        sc = {r: 0.0 for r in ROUTES}
        for r, feats in self.KW.items():
            for feat, wt in feats:
                if isinstance(feat, str) or len(feat) == 1:
                    continue                        # loose words stay quiet
                phrase = " ".join(feat)
                if phrase in (" " .join(tokenize(text))):
                    sc[r] += wt
        loud = max(sc, key=lambda r: (sc[r], -ROUTES.index(r))) \
            if any(sc.values()) else None
        if loud and sc[loud] >= LOUD.get(loud, 8):
            return loud, round(min(0.99, 0.5 + 0.05 * sc[loud]), 4), comb
        best = max(comb, key=lambda r: (comb[r], -ROUTES.index(r)))
        return best, round(comb[best], 4), comb

    # programmatic feature detectors (classic rule-based scoring, 1960s tech).
    # Phrases beat loose words: 'remind me' is evidence, 'me' alone is not.
    # Multipliers encode confidence learned the hard way (see holdout logs).
    KW = {
        "task": ((("add",), 2), (("create",), 2), (("new", "task"), 3),
                 (("task",), 1), (("todo",), 2), (("finish",), 2),
                 (("complete",), 2), (("done",), 1), (("delete",), 2),
                 (("remove",), 2), (("cancel",), 2), (("drop", "task"), 3),
                 (("postpone",), 2), (("reschedule",), 2), (("rename",), 2),
                 (("list", "tasks"), 3), (("my", "tasks"), 3),
                 (("tasks",), 1), (("plan", "day"), 3), (("goal",), 1),
                 (("start",), 1), (("begin",), 1), (("track",), 1),
                 (("jot", "down"), 3), (("gotta",), 2), (("have", "to"), 1),
                 (("need", "to"), 1), (("mark",), 1), (("check", "off"), 3),
                 (("on", "my", "plate"), 4), (("project",), 1)),
        "reminder": [(("remind",), 4), (("alarm",), 3), (("timer",), 3),
                     (("notify",), 3), (("alert",), 3), (("wake",), 2),
                     (("ping",), 2), (("nudge",), 2), (("schedule",), 2),
                     (("every", "day"), 2), (("daily",), 2), (("weekly",), 2),
                     (("hourly",), 2), (("minutes",), 1), (("pm",), 1),
                     (("am",), 1), ("o'clock", 1), (("tonight",), 1),
                     (("later",), 1), (("bell",), 2)],
        "info": [(("weather",), 4), (("forecast",), 3), (("temperature",), 3),
                 (("rain",), 2), (("snow",), 2), (("umbrella",), 3),
                 (("sunny",), 2), (("cloudy",), 2), (("search",), 3),
                 (("google",), 3), (("lookup",), 3), (("look", "up"), 4),
                 (("find",), 2), (("define",), 3), (("translate",), 3),
                 (("news",), 2), (("recipe",), 3), (("population",), 3),
                 (("explain",), 3), (("summarize",), 3), (("information",), 3),
                 (("invented",), 3), (("means",), 2), (("time",), 1),
                 (("date",), 1), (("far",), 1), (("book",), 1)],
        "system": [(("help",), 3), (("quit",), 4), (("exit",), 4),
                   (("shutdown",), 4), (("bye",), 2), (("goodbye",), 3),
                   (("status",), 2), (("memory",), 1), (("config",), 2),
                   (("reload",), 3), (("reset",), 2), (("version",), 2),
                   (("plugin",), 3), (("plugins",), 3), (("eval",), 3),
                   (("script",), 3), (("stats",), 3), (("ram",), 3),
                   (("profile",), 3), (("commands",), 3), (("enable",), 2),
                   (("disable",), 2), (("can", "you", "do"), 3),
                   (("who", "are", "you"), 4), (("your", "name"), 3),
                   (("where", "am", "i"), 3), (("set", "location"), 3),
                   (("forget", "everything"), 3), (("told", "you"), 2),
                   (("know", "about", "me"), 3), (("save", "state"), 3),
                   (("learn", "that"), 3), (("repeat", "after"), 3),
                   (("interpret",), 2), (("usage",), 2), (("located",), 3)],
        "chat": ((("hello",), 2), (("hi",), 1), (("hey",), 1), (("yo",), 1),
                 (("sup",), 2), (("sorry",), 2), (("love",), 2),
                 (("angry",), 3), (("sad",), 3), (("happy",), 3),
                 (("tired",), 2), (("afraid",), 3), (("alone",), 3),
                 (("lonely",), 3), (("bored",), 3), (("confused",), 3),
                 (("dream",), 2), (("dreams",), 2), (("mother",), 2),
                 (("father",), 2), (("friend",), 1), (("friends",), 1),
                 (("computer",), 1), (("hate",), 2), (("stressed",), 3),
                 (("sleep",), 1), (("mistakes",), 2), (("bothering",), 3),
                 (("hurts",), 3), (("sick",), 2), (("hungry",), 2),
                 (("pointless",), 3), (("matters",), 2), (("perhaps",), 2),
                 (("believe",), 2), (("decide",), 2), (("remember",), 1),
                 (("childhood",), 3), (("forgot",), 1), (("scare",), 2),
                 (("scared",), 2), (("listen",), 1), (("understands",), 3),
                 (("everyone",), 1), (("people",), 1), (("wish",), 1),
                 (("lol",), 2), (("omg",), 2), (("brb",), 3), (("g2g",), 3),
                 (("thx",), 3), (("ur",), 2), (("idk",), 3), (("wbu",), 3),
                 (("vibing",), 3), (("bro",), 2), (("cap",), 1), (("bet",), 1),
                 (("cool",), 1), (("funny",), 2), (("joke",), 2),
                 (("useless",), 2), (("robot",), 2), (("machine",), 1),
                 (("real",), 1), (("morning",), 1), (("giving", "up"), 3),
                 (("feel",), 2), (("feeling",), 2), (("thinking",), 1),
                 (("annoyed",), 3), (("frustrated",), 3), (("anxious",), 3),
                 (("excited",), 2), (("grateful",), 2), (("hopeless",), 3)),
    }

    def knowledgeable_scores(self, text):
        toks = tokenize(text)
        joined = " ".join(toks)
        bigrams = {"%s %s" % pair for pair in zip(toks, toks[1:])}
        trigrams = {"%s %s %s" % trip for trip in
                    zip(toks, toks[1:], toks[2:])}
        sc = {r: 0.0 for r in ROUTES}
        for r, feats in self.KW.items():
            for feat, w in feats:
                if isinstance(feat, str):          # single token
                    if feat in toks:
                        sc[r] += w
                elif len(feat) == 1:
                    if feat[0] in toks:
                        sc[r] += w
                elif len(feat) == 2:
                    if feat[0] in toks and feat[1] in toks:
                        sc[r] += w
                else:                              # exact phrase
                    if " ".join(feat) in joined or " ".join(feat) in bigrams \
                            or " ".join(feat) in trigrams:
                        sc[r] += w
        tot = sum(sc.values())
        if tot == 0:
            return {r: 1.0 / len(ROUTES) for r in ROUTES}
        k = 0.4
        return {r: (v + k) / (tot + k * len(ROUTES)) for r, v in sc.items()}

    # -------------------------------------------------------- online learning
    def reinforce(self, text, route, weight=1):
        """Perceptron-ish online update: user corrections teach the web.
        (Arthur Samuel 1959: machines that learn without being explicitly
        programmed. Our version fits in a function.)"""
        if not self.w3:
            return
        for w in content_words(text):
            i = self.vocab.get(w)
            if i is None:
                continue
            j = ROUTES.index(route)
            self.counts[i][j] += weight
            self._dirty = True
        if self._dirty:
            self._rebuild_w3()
            self.save()

    # --------------------------------------------------------- semantic glue
    def similarity(self, a, b):
        """Word-web cosine over ternary route evidence + optional WordNet."""
        if self.lex is not None:
            s = self.lex.similarity(a, b, "wup")
            if s is not None and s > 0:
                return round(float(s), 3)
        ia, ib = self.vocab.get(a), self.vocab.get(b)
        if ia is None or ib is None or not self.w3:
            return 0.0
        va, vb = self.w3[ia], self.w3[ib]
        num = sum(x * y for x, y in zip(va, vb))
        da = math.sqrt(sum(x * x for x in va)) or 1.0
        db = math.sqrt(sum(y * y for y in vb)) or 1.0
        return round(num / (da * db), 3)

    def nearest(self, word, k=5):
        if word not in self.vocab or not self.w3:
            return []
        i = self.vocab[word]
        scored = []
        for w, j in self.vocab.items():
            if w == word:
                continue
            s = self.similarity(word, w)
            if s > 0:
                scored.append((s, w))
        scored.sort(reverse=True)
        return scored[:k]

    def stats(self):
        return {"vocab": len(self.words), "routes": ROUTES,
                "corpus": self.meta.get("corpus_size"),
                "mlp_acc": self.meta.get("mlp_train_acc"),
                "has_net": bool(self.net), "numpy": np is not None}
