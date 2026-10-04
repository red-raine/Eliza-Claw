"""Build the WordWeb: vocabulary + ternary Bayesian matrix + backprop MLP.

Usage:  python -m elizaclaw.tools.build_wordweb [--corpus data/routes.jsonl]

This is the 'training time' half. Runtime (shell.py) only loads the JSON.
Zero tokens: numpy + WordNet + arithmetic, no pretrained embeddings.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from elizaclaw.wordweb import WordWeb, build_vocab, ROUTES  # noqa: E402


def load_corpus(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def holdout_eval(web, corpus, frac=0.25, seed=3):
    """Scientific method: measure, don't vibe. Score the ensemble on a
    deterministic random holdout it never saw during this final pass."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(corpus)))
    rng.shuffle(idx)
    cut = int(len(idx) * frac)
    test = [corpus[i] for i in idx[:cut]]
    ok = tot = 0
    per = {r: [0, 0] for r in ROUTES}
    conf_ok = conf_all = 0.0
    for ex in test:
        pred, conf, _ = web.route(ex["text"])
        tot += 1
        per[ex["label"]][1] += 1
        if pred == ex["label"]:
            ok += 1
            per[ex["label"]][0] += 1
            conf_ok += conf
        conf_all += conf
    return {"n_test": tot, "accuracy": round(ok / max(tot, 1), 4),
            "avg_conf_correct": round(conf_ok / max(ok, 1), 4),
            "avg_conf_all": round(conf_all / max(tot, 1), 4),
            "per_class": {r: "%d/%d" % (ok_, n) for r, (ok_, n) in
                          per.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/routes.jsonl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=250)
    args = ap.parse_args()

    lex = None
    try:
        from elizaclaw.lexicon import Lexicon
        lex = Lexicon(os.path.join(args.out, "..", "memory"))
    except Exception:
        pass

    corpus = load_corpus(args.corpus)
    print("corpus:", len(corpus), "examples")
    vocab, words = build_vocab(corpus, lex=lex)
    print("vocab:", len(words), "terms (word-web grown with lemmas+conceptnet)")

    web = WordWeb(root=args.out, lex=lex)
    web.vocab, web.words = vocab, words
    web.fit_stats(corpus)
    web.train_mlp(corpus, hidden=args.hidden, epochs=args.epochs)
    web.meta["routes"] = ROUTES
    report = holdout_eval(web, corpus)
    web.meta["holdout"] = report
    web.save()
    print("model saved to", os.path.join(args.out, "wordweb.json"),
          "(%.1f KB)" % (os.path.getsize(
              os.path.join(args.out, "wordweb.json")) / 1024))
    print("holdout:", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
