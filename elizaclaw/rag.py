"""Eliza-Claw :: rag -- retrieval-augmented generation WITHOUT a language model.

Salton's 1970s SMART TF-IDF cosine, in ~150 lines. Documents come from three
feeds: (1) the curated topic corpus (data/topics.json), (2) episodic + mid
term memory, (3) web-search snippets pushed by plugins/websearch.py. The
retrieved passage is slotted into Mad-Lib templates -- "generation" here is
template rendering, exactly as ELIZA did it; we just added a bookshelf.

numpy when available for the vector math, pure-python fallback otherwise.
Zero tokens, zero embeddings, inspectable scores.
"""
import json
import math
import os
import re

try:
    import numpy as _np
except ImportError:      # pragma: no cover - optional dep
    _np = None

HERE = os.path.dirname(os.path.abspath(__file__))
STOPWORDS_FILE = os.path.join(HERE, "..", "data", "common_words.json")


def _default_stopwords():
    try:
        with open(STOPWORDS_FILE) as f:
            data = json.load(f)
        return {w["word"] for w in data["words"] if w.get("stop")}
    except Exception:
        return {"the", "a", "an", "is", "are", "was", "were", "of", "to",
               "in", "on", "and", "or", "i", "you", "it", "this", "that"}


_TOKEN = re.compile(r"[a-z0']+")


class Rag:
    def __init__(self, stopwords=None):
        self.stop = stopwords or _default_stopwords()
        self.docs = []          # [{id, text, meta}]
        self.df = {}            # term -> doc frequency
        self.vecs = []          # per-doc term->tf count dicts
        self.idf = {}
        self._dirty = False

    # ------------------------------------------------------------------ utils
    def tokenize(self, text):
        return [t for t in _TOKEN.findall(text.lower()) if t not in self.stop]

    # ------------------------------------------------------------------ index
    def add(self, doc_id, text, meta=None):
        """Idempotent upsert by doc_id."""
        for i, d in enumerate(self.docs):
            if d["id"] == doc_id:
                self.docs[i] = {"id": doc_id, "text": text, "meta": meta or {}}
                self._dirty = True
                return
        self.docs.append({"id": doc_id, "text": text, "meta": meta or {}})
        self._dirty = True

    def remove(self, doc_id):
        self.docs = [d for d in self.docs if d["id"] != doc_id]
        self._dirty = True

    def _rebuild(self):
        self.df, self.vecs = {}, []
        for d in self.docs:
            toks = self.tokenize(d["text"])
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.vecs.append(tf)
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        n = max(1, len(self.docs))
        self.idf = {t: math.log((n + 1) / (dfv + 0.5)) for t, dfv in self.df.items()}
        if _np is not None and self.vecs:
            vocab = sorted(self.idf)
            self._mat = _np.zeros((len(self.vecs), len(vocab)), dtype="float32")
            self._vindex = {t: i for i, t in enumerate(vocab)}
            for r, tf in enumerate(self.vecs):
                for t, c in tf.items():
                    self._mat[r, self._vindex[t]] = c * self.idf[t]
            norms = _np.linalg.norm(self._mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._mat /= norms
        else:
            self._mat = None
        self._dirty = False

    # ----------------------------------------------------------------- query
    def search(self, query, k=3):
        """Return [(score, doc)] top-k, cosine TF-IDF."""
        if self._dirty:
            self._rebuild()
        if not self.docs:
            return []
        toks = self.tokenize(query)
        if not toks:
            return []
        qtf = {}
        for t in toks:
            qtf[t] = qtf.get(t, 0) + 1
        qvec = {t: c * self.idf.get(t, math.log(len(self.docs))) for t, c in qtf.items()}
        if self._mat is not None:
            v = _np.zeros(self._mat.shape[1], dtype="float32")
            for t, w in qvec.items():
                idx = self._vindex.get(t)
                if idx is not None:
                    v[idx] = w
            nrm = _np.linalg.norm(v)
            if nrm:
                v /= nrm
            sims = self._mat @ v
            order = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
            return [(float(sims[i]), self.docs[i]) for i in order if sims[i] > 0]
        # pure-python fallback
        qnorm = math.sqrt(sum(w * w for w in qvec.values())) or 1.0
        scored = []
        for d, tf in zip(self.docs, self.vecs):
            dot = 0.0
            for t, w in qvec.items():
                if t in tf:
                    dot += w * tf[t] * self.idf.get(t, 0)
            dn = math.sqrt(sum((c * self.idf.get(t, 0)) ** 2
                               for t, c in tf.items())) or 1.0
            s = dot / (qnorm * dn)
            if s > 0:
                scored.append((s, d))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def best(self, query):
        hits = self.search(query, k=1)
        return hits[0][1] if hits else None

    def load_corpus(self, path=None):
        """Index topics.json follow-up passages as retrievable documents."""
        path = path or os.path.join(HERE, "..", "data", "topics.json")
        with open(path) as f:
            tp = json.load(f)
        for t in tp["topics"]:
            body = (f"{t['name']} ({t['domain']}). People talk about: "
                    + ", ".join(t["aliases"]) + ".")
            self.add(f"topic:{t['id']}", body, meta={"kind": "topic",
                                                     "name": t["name"],
                                                     "topic": t})
        return len(tp["topics"])

    def sync_memory(self, tiers):
        """Feed MTM events + episodes + facts into the index each tick."""
        cnt = 0
        for ev in tiers.mtm[-50:]:
            self.add(f"mtm:{ev['t']:.0f}", ev["text"], meta={"kind": "mtm"})
            cnt += 1
        for e in tiers.epi[-50:]:
            self.add(f"epi:{e['t']:.0f}",
                     f"When we talked about {e.get('topic','life')}: {e['what']}",
                     meta={"kind": "episode"})
            cnt += 1
        for k, v in tiers.facts().items():
            self.add(f"ltm:{k}", f"Fact: {v}", meta={"kind": "fact"})
            cnt += 1
        return cnt

    def stats(self):
        return {"docs": len(self.docs), "terms": len(self.df),
                "backend": "numpy" if self._mat is not None else "python"}
