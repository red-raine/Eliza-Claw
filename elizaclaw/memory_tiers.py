"""Eliza-Claw :: memory_tiers -- STM / MTM / LTM / Episodic (Wave 1).

Four tiers, one module, all JSON-backed via state.py. Lineage: Atkinson-
Shiffrin (1968) multi-store + ACT-R (Anderson '80s) activation decay +
Tulving episodic/semantic split + Cornell robot-episodic-memory (Ziarko '17)
valence tagging.

  STM  short-term : in-RAM ring buffer of the current session's turns
  MTM  mid-term   : last N sessions persisted; salience-weighted recall
  LTM  long-term  : durable facts (key -> value), never decays, only consolidated
  EPI  episodic   : timestamped, valence-tagged moments ("user was sad about X")

Decay runs on the heartbeat tick: activation *= exp(-lambda * dt); items below
tau get promoted to LTM if repeatedly hot (consolidation), else forgotten.
Zero tokens, zero tensors -- just dicts and math.
"""
import math
import time


class MemoryTiers:
    STM_CAP = 40          # turns held in RAM
    MTM_KEEP = 200        # persisted recent events
    DECAY_LAMBDA = 0.05   # per-minute activation decay (ACT-R-ish)
    FORGET_TAU = 0.05     # below this and cold => prune from MTM
    CONSOLIDATE_HITS = 3  # recalled >=N times => promote to LTM

    def __init__(self, state):
        self.state = state
        self.stm = []                                   # [{role,text,t}]
        m = state.load("memory_tiers", {"mtm": [], "ltm": {}, "epi": [],
                                        "hits": {}})
        self.mtm = m["mtm"]      # [{kind,text,t,activation,hits}]
        self.ltm = m["ltm"]      # {key: {text, t}}
        self.epi = m["epi"]      # [{what, who, valence, topic, t}]
        self.hits = m.get("hits", {})

    # ------------------------------------------------------------- persistence
    def _persist(self):
        self.state.save("memory_tiers", {"mtm": self.mtm, "ltm": self.ltm,
                                         "epi": self.epi, "hits": self.hits})

    # ------------------------------------------------------------------- write
    def add_turn(self, role, text):
        """Every user/bot turn lands in STM; a summary event into MTM."""
        now = time.time()
        self.stm.append({"role": role, "text": text, "t": now})
        if len(self.stm) > self.STM_CAP:
            self.stm.pop(0)
        self.mtm.append({"kind": "turn", "text": f"{role}: {text}",
                         "t": now, "activation": 1.0, "hits": 0})
        if len(self.mtm) > self.MTM_KEEP:
            self.mtm.pop(0)
        self._persist()

    def remember_fact(self, key, text):
        """Direct LTM write (explicit 'remember that...' commands)."""
        self.ltm[key] = {"text": text, "t": time.time()}
        self._persist()

    def record_episode(self, what, topic="", valence=0.0, who="user"):
        """Episodic tag: what happened, under which topic, how it felt.
        valence in [-1, 1] comes from lexicon.sentiment()."""
        self.epi.append({"what": what, "who": who, "topic": topic,
                         "valence": round(valence, 3), "t": time.time(),
                         "activation": 1.0})
        if len(self.epi) > 500:
            self.epi.pop(0)
        self._persist()

    # ------------------------------------------------------------------- read
    def recent(self, n=6):
        return self.stm[-n:]

    def recall_mtm(self, query_tokens, k=3):
        """Salience-weighted substring/token recall over MTM events."""
        scored = []
        qset = set(query_tokens)
        for ev in self.mtm:
            toks = set(ev["text"].lower().split())
            overlap = len(qset & toks)
            if overlap:
                scored.append((overlap * ev["activation"], ev))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, ev in scored[:k]:
            ev["hits"] = ev.get("hits", 0) + 1
            ev["activation"] = min(1.0, ev["activation"] + 0.2)  # reheat
            self.hits[ev["text"]] = self.hits.get(ev["text"], 0) + 1
            # consolidation: hot + repeated => promote to LTM
            if ev["hits"] >= self.CONSOLIDATE_HITS:
                self.ltm.setdefault(f"hot:{ev['text'][:32]}",
                                    {"text": ev["text"], "t": time.time()})
            out.append(ev["text"])
        if out:
            self._persist()
        return out

    def episodes(self, topic=None, k=5):
        eps = [e for e in self.epi if not topic or e.get("topic") == topic]
        eps.sort(key=lambda e: -(e.get("activation", 1.0) * 1e9 + e["t"]))
        return eps[:k]

    def facts(self):
        return {k: v["text"] for k, v in self.ltm.items()}

    # ------------------------------------------------------------------ tick
    def decay(self, minutes=1.0):
        """Heartbeat hook: ACT-R base-level decay on MTM + episodic."""
        f = math.exp(-self.DECAY_LAMBDA * minutes)
        for ev in self.mtm:
            ev["activation"] *= f
        for e in self.epi:
            e["activation"] = e.get("activation", 1.0) * f
        # prune dead weight (but never LTM)
        before = len(self.mtm)
        self.mtm = [e for e in self.mtm
                    if e["activation"] > self.FORGET_TAU or e["hits"] > 0]
        if len(self.mtm) != before:
            self._persist()

    def stats(self):
        return {"stm": len(self.stm), "mtm": len(self.mtm),
                "ltm": len(self.ltm), "episodic": len(self.epi)}
