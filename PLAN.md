# Eliza-Claw :: Master Plan (honest edition)

> Zero tokens. One loop. A heartbeat. Everything below is either DONE (with
> passing tests), IN PROGRESS, or PLANNED. No hallucinated waves — if it's
> not tested, it doesn't get a checkmark.

## Where we actually are (verified 2026-10-05)

Core modules that exist and run: `shell.py`, `nlu.py` (DP pattern compiler,
COMPASS semantics), `lexicon.py` (WordNet/synsets), `wordweb.py` (Bayesian
ternary router + tiny MLP), `state.py` (JSON memory), `plugin.py`, `clan.py`,
plugins: identity / eliza_conversation / task_manager / tools / system / clan.
Smoke test passes (`echo hello | python -m elizaclaw`). ~3,400 LOC total.

---

## Wave 0 — Foundation & Pattern Engine ✅ DONE
- [x] load → parse → route → execute → template → respond loop
- [x] Plugin base class + auto-registration
- [x] JSON state manager (memory/, per-plugin state, history log)
- [x] Heartbeat tick with plugin wake-ups
- [x] compile_pattern rebuilt with dynamic programming (fixed \x08 backspace
      corruption, leading-star hijack, interior-star empty tail, capture drift)
- [x] WordNet synonym expansion into @syn sets (lexicon.expand_synset)
- [x] Bayesian WordWeb router (ternary weights, numpy when available)

## Wave 1 — Conversation Corpus + Tiered Memory + RAG 🚧 THIS WAVE
The user's brief: *"1-2k of the most common human discussions and topics
discussed online… deep conversations, follow up replies, long/short/mid term,
episodic memory and a RAG it uses, plus web search with high quality fact
checking."* Sources mined (offline-safe): Wikipedia Most Common Words in
English, frequencylist.com top-2000, psycatgames/socialself/pleasant-
conversation topic lists, classic ELIZA script domains.

- [ ] `data/common_words.json` — top ~2000 English function/content words
      (stopwords + non-stopwords flagged) built from NLTK stopwords ∪ word freq
- [ ] `data/topics.json` — ~1,000–2,000 conversation topics/phrases/slang,
      each with: canonical name, aliases, small-talk opener templates,
      **follow-up question chains** (deep-conversation ladders à la Arthur Aron's
      36 questions), sentiment tag, domain bucket
- [ ] `elizaclaw/memory_tiers.py` — STM (session buffer), MTM (last N turns +
      salience), LTM (persistent JSON), Episodic (timestamped moments with
      emotional-valence tags — Ziarko/episodic-memory lineage)
- [ ] `elizaclaw/rag.py` — zero-token retrieval: TF-IDF cosine over topics +
      episodic store, no embeddings, numpy scoring; retrieves passages to fill
      Mad-Lib templates
- [ ] `plugins/conversation.py` — topic detection via WordWeb route, opener
      pick, follow-up chain walker (asks deeper questions while user stays
      engaged), ties into memory tiers + RAG
- [ ] `plugins/websearch.py` — DuckDuckGo Lite HTML scrape (stdlib urllib, no
      API key, no LLM), result snippets into RAG index
- [ ] `plugins/factcheck.py` — multi-source agreement scoring: ≥2 independent
      snippets agree on extracted entity/number ⇒ "high confidence"; else
      hedge like a good 1966 ELIZA would
- [ ] `tests/test_wave1.py` — corpus loads, tier round-trips, RAG retrieval
      sanity ("tell me about dogs" retrieves dog passage), follow-up chain
      advances, factcheck abstains offline
- [ ] Fix leftover: pytest marker bug in tmp/wave0_verify (move real tests to
      tests/, delete scratch)

## Wave 2 — Deep Dialogue Mechanics 📅 PLANNED
- [ ] Goal-directed dialogue stacks (Winograd-style goals, Schank scripts)
- [ ] Mixed-initiative clarification (when confidence < τ, ask instead of act)
- [ ] Persona consistency via frames (Minsky) — slot values persist across days
- [ ] Discourse markers & pronoun resolution coreference-lite (spaCy optional)

## Wave 3 — Eval & Beat-the-Parrot 📅 PLANNED
- [ ] Intent accuracy suite (unseen paraphrases, slot F1) — deterministic
- [ ] Ablations: worldnet off / bayes off / stemmer off / pattern-only
- [ ] Side-by-side punchcard vs GPT-4 on narrow tasks (reminders, routing)
- [ ] Footprint report: LOC, MB, startup ms (smaller = more points)

## Wave 4 — Micro OS & Automation 📅 PLANNED
- [ ] ELF'83 scripting (Lisp+Prolog+Smalltalk+Logo+CLIPS mashup) — clan.py seed exists
- [ ] Screen macros / desktop automation without vision models (xdotool-style keys)
- [ ] MUD-bot lineage: trigger tables + alias language (GMCP/MXP ideas)

## Wave 5 — Docs, Wiki, Logo 📅 PLANNED
- [ ] llm_wiki/ code walkthrough (what & why per module)
- [ ] research/ decade-by-decade notes cross-referenced from PLAN
- [ ] AGENTS.md usage guide, cute claw logo (SVG, pure text)

### Cross-references
- Research sources logged in `research/SOURCES.md`
- Decisions/experiments logged in `llm_wiki/DECISIONS.md`
