# Eliza-Claw :: Research Sources & Decade Mining Log

Cross-reference: `../PLAN.md` (waves), `../llm_wiki/DECISIONS.md` (choices).

## Wave 1 corpus sources (user-provided, mined offline-safe)
| Source | What we took | Where it lands |
|---|---|---|
| Wikipedia "Most common words in English" (Oxford 5000) | top ~2000 word list w/ POS-ish buckets | data/common_words.json |
| frequencylist.com top-2000 | frequency ranks, stopword flags | data/common_words.json |
| psycatgames conversation starters | deep-question ladders | data/topics.json follow_up chains |
| socialself "interesting things to talk about" | topic buckets (travel, food, movies…) | data/topics.json domains |
| scribd 1000 daily-conversation topics | 1000+ canonical topic names | data/topics.json names |
| thepleasantconversation topics | small-talk openers per topic | data/topics.json openers |
| NLTK stopwords + WordNet | stopword set, synonym growth | lexicon.py @syn expansion |

## Decade mining (history of AI → features)
- **1950s** Turing test, Dartmouth. Idea: fool-the-judge evals -> Wave 3 punchcard vs GPT-4.
- **1960s** ELIZA (Weizenbaum '66), COMPASS pattern matcher (Colby '65: *
      wildcards + prepositional-phrase decomposition) -> nlu.py DP compiler IS
      COMPASS semantics. PARRY '72 paranoia model -> sentiment tags on topics.
- **1970s** Minsky frames '75 -> frame slots in lexicon.fill_frame; Schank
      scripts -> Wave 2 goal stacks; SHRDLU blocks world -> ELF'83 turtle.
- **1980s** Prolog/Lisp machines, CLIPS rules, LPMud LPC bots -> clan.py +
      Wave 4 scripting/MUD triggers. ACT-R memory activation -> memory_tiers
      salience decay.
- **1990s** ALICE AIML (that/kind/category recursion) -> eliza.json keys;
      TF-IDF (Salton legacy) -> rag.py scoring; WordNet 2.0 -> lexicon.
- **2000s** Wikipedia search snippets, DuckDuckGo Instant Answer era ->
      websearch/factcheck multi-source agreement (no LLM, just overlap).
- **2010s** Episodic memory robots (Ziarko Cornell '17) -> episodic tier with
      valence tags; BERT embeddings -- deliberately NOT used (zero tokens);
      pocketflow-style graph agents -> heartbeat queue replaces DAG.
- **2020s** LLM agents (ReAct, AutoGPT) = our foil. OpenClaw/TinyClaw/Hermes
      agent analysis: their loop is load->route->tool->remember; we do all four
      without a model. RAG is standard -- ours is TF-IDF cosine, no vectors.

## Fact-checking without an LLM (research notes)
Multi-source agreement: query DuckDuckGo Lite, take top-k snippets, extract
numbers/entities with regex+NER-lite, count cross-document agreement.
>=2 independent domains agreeing => high confidence; else hedge ("I could be
mistaken, but..."). Classical IR truthiness work (Passonneau lineage)
shrunk to stdlib.

## Why no embeddings / no LLM (challenge rules)
Rule 2: zero tokens. Rule 4: smaller footprint = points. TF-IDF cosine over a
curated 2k-topic corpus retrieves as well as tiny embeddings for narrow
chit-chat routing, at 0 MB of model weights. The parrot cannot compete on
footprint.
