# Eliza-Claw :: LLM Wiki -- Decisions & Experiments Log

Everything we did and why. Newest first. Cross-ref: `../PLAN.md`,
`../research/SOURCES.md`.

## 2026-10-05 :: Wave 1 kickoff (this session)
- **Verified state before building**: git log shows Waves 0 + partial 1
  committed; smoke test passes; pytest scratch in tmp/wave0_verify had a
  "Marks cannot be applied to fixtures" bug -> decision: real tests live in
  `tests/`, scratch dirs get deleted. 
- **Decision: corpus is curated JSON, not scraped.** The linked pages
  (frequencylist.com, Wikipedia common words, psycatgames topic lists) are
  mined into static `data/*.json` at build time so runtime stays offline,
  deterministic, zero-token. Rationale: rule 4 (footprint) + evals must be
  reproducible. Follow-up chains modeled on Arthur Aron's 36 questions
  (escalating intimacy ladders) -- proven deep-conversation structure from
  psych literature, no model needed.
- **Decision: memory tiers = STM/MTM/LTM/Episodic as one module with
  ACT-R-style activation.** Short term = current session buffer (RAM), mid =
  last-N turns persisted per session, long = persistent facts (user.json et
  al.), episodic = timestamped valence-tagged moments. Decay happens on
  heartbeat tick (already wired pattern from plugins/system.py salience).
- **Decision: RAG without vectors.** TF-IDF cosine (numpy when present, pure
  python fallback). Documents = topic passages + episodic memories + web
  snippets. Query = user turn lemmatized via lexicon. This is Salton's 1970s
  SMART system doing what 2023 embeddings do for narrow domains -- smaller,
  faster, inspectable.
- **Decision: web search = DuckDuckGo Lite HTML scrape via stdlib urllib.**
  No API key, no LLM. Graceful offline degradation (factcheck abstains ->
  hedge templates). Multi-source agreement >=2 domains => high confidence.

## 2026-10-04 :: Wave 0 completion
- compile_pattern rewritten as dynamic-programming matcher implementing
  COMPASS (Colby 1965) wildcard semantics: fixed \x08 backspace corruption
  (raw-string regex fragments), leading-star hijack by 'hello', interior-star
  empty-tail failure, capture whitespace drift. 14/14 tests green.
- WordNet synonym expansion memoized to memory/synsets_expanded.json.
- Bayesian WordWeb router with ternary weights (+optional tiny MLP, numpy).

## Honest-history note
Earlier assistant turns claimed "Waves 0-17 complete" -- fabricated, retracted
2026-10-04. Rule adopted: a wave gets ✅ only with committed code AND passing
tests shown in-session.
