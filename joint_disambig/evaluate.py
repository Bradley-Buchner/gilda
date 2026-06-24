"""Evaluation and feasibility check for joint disambiguation model results
"""
import os
import pickle
import sys
from collections import defaultdict
from itertools import product
from typing import Optional
import numpy as np
import pandas as pd

import gilda
from .data import DocumentExample
from .embedder import CandidateEmbedder

# Path to dir with old network-based joint disamb models
_NETWORK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, "network",
)


def feasibility_check(
    embedder: CandidateEmbedder,
    deepwalk_model_path: Optional[str] = None,
    test_cases: Optional[list[list[str]]] = None,
):
    """Test whether PubMedBERT-derived embeddings for Gilda candidates actually
    provide signal for disambiguation based on a cosine similarity assessment.
    For each test case, the check embeds all Gilda candidates, computes
    cross-mention embedding cosine similarities, and finds the candidate combo
    with the best coherence (most similarity). You can optionally run this check
    with DeepWalk embeddings to compare the two side-by-side.
    """
    if test_cases is None:
        test_cases = [
            ["ER", "tamoxifen", "PR"],
            ["BRCA1", "breast", "p53"],
        ]

    dw_kv, dw_vocab, dw_vectors = None, None, None
    if deepwalk_model_path:
        with open(deepwalk_model_path, "rb") as f:
            dw_kv = pickle.load(f)
        dw_vocab = dw_kv.__dict__.get("vocab", {})
        dw_vectors = dw_kv.vectors

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test case: {case}")
        print(f"{'='*60}")

        mention_candidates = {}
        for text in case:
            matches = gilda.ground(text)
            mention_candidates[text] = matches[:5]

        llm_embs = {}
        for text, matches in mention_candidates.items():
            for i, m in enumerate(matches):
                llm_embs[(text, i)] = embedder.embed_candidate(m.term)

        for text, matches in mention_candidates.items():
            print(f"\n  {text}:")
            for i, m in enumerate(matches):
                print(f"    [{i}] {m.term.entry_name} ({m.term.db}:{m.term.id}) "
                      f"score={m.score:.3f}")

        print(f"\n  --- LLM pairwise cosine similarities ---")
        _print_coherence_matrix(case, mention_candidates, llm_embs)

        if dw_kv is not None:
            dw_embs = {}
            for text, matches in mention_candidates.items():
                for i, m in enumerate(matches):
                    node = (m.term.id if m.term.db in ("CHEBI", "GO")
                            else m.term.entry_name)
                    if node in dw_vocab:
                        idx = dw_vocab[node].index
                        dw_embs[(text, i)] = dw_vectors[idx]
            if dw_embs:
                print(f"\n  --- DeepWalk pairwise cosine similarities ---")
                _print_coherence_matrix(case, mention_candidates, dw_embs)
            else:
                print("\n  (No DeepWalk embeddings found for these candidates)")


def _print_coherence_matrix(case, mention_candidates, embs):
    """Print cross-mention cosine similarity values for all candidate pairs.
    """
    mentions = list(case)
    if len(mentions) < 2:
        return
    for i, t1 in enumerate(mentions):
        for t2 in mentions[i + 1:]:
            print(f"\n    {t1} x {t2}:")
            cands1 = mention_candidates[t1]
            cands2 = mention_candidates[t2]
            best_sim, best_pair = -1, None
            for a, c1 in enumerate(cands1):
                for b, c2 in enumerate(cands2):
                    e1 = embs.get((t1, a))
                    e2 = embs.get((t2, b))
                    if e1 is None or e2 is None:
                        continue
                    sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) *
                                                  np.linalg.norm(e2) + 1e-9))
                    print(f"      {c1.term.entry_name} <-> {c2.term.entry_name}: "
                          f"{sim:.4f}")
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (c1.term.entry_name, c2.term.entry_name)
            if best_pair:
                print(f"    Best: {best_pair[0]} <-> {best_pair[1]} "
                      f"({best_sim:.4f})")


def _doc_to_candidate_dicts(doc: DocumentExample) -> dict[str, list[dict]]:
    """Convert DocumentExample mentions to dictionary format used by PPR/cosine
    code.
    """
    candidates = {}
    for m in doc.mentions:
        text_cands = []
        for cand in m.candidates:
            db = cand.term.db
            node = cand.term.id if db in ("CHEBI", "GO") else (
                cand.term.entry_name)
            text_cands.append({
                "node": node,
                "db": db,
                "id": cand.term.id,
                "curie": f"{db}:{cand.term.id}",
                "gilda_score": cand.score,
            })
        candidates[m.text] = text_cands
    return candidates


def evaluate_ppr(
    docs: list[DocumentExample],
    graph_path: str,
    alpha: float = 0.15,
    beta: float = 0.5,
    norm_method: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate PPR-based joint disambiguation on a corpus of documents.

    Params:
    ------
    docs :
        List of DocumentExample instances to use for evaluation.
    graph_path :
        Path to pickled multigraph used by PPR
    alpha :
        PPR damping factor that controls the probability of restarting the
        random walk from the personalization vector.
    beta :
        Mixing weight between normalized PPR score and Gilda lexical score
        used when re-ranking candidates.
    norm_method :
        Method for normalizing PPR scores. Must be one of 'divide', 'log',
        'sqrt', 'global_ratio', 'global_diff', or None.

    Returns:
    --------
    pd.DataFrame :
        Evaluation results grouped by entity type.
    """
    sys.path.insert(0, os.path.dirname(_NETWORK_DIR))
    from network.ppr_disambiguation import (
        load_graph, multigraph_to_simple, build_personalization_vector,
        run_ppr, normalize_ppr, select_groundings, compute_global_pagerank,
    )

    print("Loading graph for PPR baseline...")
    mg = load_graph(graph_path)
    graph = multigraph_to_simple(mg)

    global_pagerank = None
    if norm_method in ("global_diff", "global_ratio"):
        print("Computing global PageRank...")
        global_pagerank = compute_global_pagerank(graph, alpha=alpha)

    counts = defaultdict(lambda: {"correct": 0, "has_grounding": 0,
                                  "total": 0})

    for doc in docs:
        candidates = _doc_to_candidate_dicts(doc)
        filtered = {}
        fallbacks = {}
        for text, cands in candidates.items():
            in_graph = [c for c in cands if c["node"] in graph]
            if in_graph:
                filtered[text] = in_graph
            elif cands:
                fallbacks[text] = cands[0]["curie"]
            else:
                fallbacks[text] = None

        ppr_results = {}
        if filtered:
            pv = build_personalization_vector(filtered, graph)
            ppr_scores = run_ppr(graph, pv, alpha=alpha)
            norm_scores = normalize_ppr(
                ppr_scores, graph, method=norm_method,
                global_pagerank=global_pagerank,
            )
            groundings = select_groundings(filtered, ppr_scores,
                                           beta=beta, normalized_scores=norm_scores)
            for text, best in groundings.items():
                if best is not None:
                    ppr_results[text] = best["curie"]

        # Evaluate
        for mention in doc.mentions:
            etype = mention.entity_type
            counts[etype]["total"] += 1
            top_curie = ppr_results.get(mention.text) or fallbacks.get(mention.text)
            if top_curie is None and mention.candidates:
                top_curie = (f"{mention.candidates[0].term.db}:"
                             f"{mention.candidates[0].term.id}")
            if top_curie is None:
                continue
            counts[etype]["has_grounding"] += 1
            if top_curie in mention.gold_synonyms:
                counts[etype]["correct"] += 1

    return _counts_to_df(counts)


def evaluate_deepwalk_cosine(
    docs: list[DocumentExample],
    model_path: str,
    cosine_alpha: float = 0.5,
) -> pd.DataFrame:
    """Evaluate the DeepWalk-based method for joint disambiguation.

    Params:
    -------
    docs :
        List of DocumentExample instances to use for evaluation.
    model_path :
        Path to pickled DeepWalk KeyedVectors.
    cosine_alpha :
        Mixing weight that controls the balance between relying on embedding
        coherence or original Gilda scores.

    Returns:
    --------
    pd.DataFrame :
        Evaluation results grouped by entity type.
    """
    with open(model_path, "rb") as f:
        kv = pickle.load(f)
    vocab = kv.__dict__.get("vocab", {})
    vectors = kv.vectors

    def _get_vec(key):
        if key in vocab:
            return vectors[vocab[key].index]
        return None

    def _cosine(v1, v2):
        d = np.dot(v1, v2)
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        return d / n if n > 0 else 0.0

    counts = defaultdict(lambda: {"correct": 0, "has_grounding": 0,
                                  "total": 0})

    for doc in docs:
        candidates = _doc_to_candidate_dicts(doc)
        resolved = {}  # text: curie
        ambiguous = {}  # text: [node, curie, gilda_score, vec]
        for text, cands in candidates.items():
            embeddable = []
            for c in cands:
                vec = _get_vec(c["node"])
                if vec is not None:
                    embeddable.append((c["node"], c["curie"],
                                       c["gilda_score"], vec))
            if len(embeddable) <= 1:
                if embeddable:
                    resolved[text] = embeddable[0][1]
                elif cands:
                    resolved[text] = cands[0]["curie"]
            else:
                ambiguous[text] = embeddable

        if ambiguous:
            amb_texts = list(ambiguous.keys())
            cand_lists = [ambiguous[t] for t in amb_texts]
            n_combos = 1
            for cl in cand_lists:
                n_combos *= len(cl)
            if n_combos > 100000:
                for t in amb_texts:
                    resolved[t] = ambiguous[t][0][1]
            else:
                best_score, best_combo = -float("inf"), None
                res_vecs = []
                res_gilda = []
                for t, curie in resolved.items():
                    for c in candidates[t]:
                        if c["curie"] == curie:
                            v = _get_vec(c["node"])
                            if v is not None:
                                res_vecs.append(v)
                                res_gilda.append(c["gilda_score"])
                            break

                for combo in product(*cand_lists):
                    all_vecs = res_vecs + [c[3] for c in combo]
                    all_gilda = res_gilda + [c[2] for c in combo]
                    n = len(all_vecs)
                    avg_cos = 0.0
                    if n >= 2:
                        total = sum(
                            _cosine(all_vecs[i], all_vecs[j])
                            for i in range(n) for j in range(i+1, n)
                        )
                        avg_cos = total / (n * (n-1) / 2)
                    score = (cosine_alpha * avg_cos +
                             (1 - cosine_alpha) * np.mean(all_gilda))
                    if score > best_score:
                        best_score = score
                        best_combo = combo

                if best_combo:
                    for t, chosen in zip(amb_texts, best_combo):
                        resolved[t] = chosen[1]

        # Evaluate
        for mention in doc.mentions:
            etype = mention.entity_type
            counts[etype]["total"] += 1
            top_curie = resolved.get(mention.text)
            if top_curie is None and mention.candidates:
                top_curie = (f"{mention.candidates[0].term.db}:"
                             f"{mention.candidates[0].term.id}")
            if top_curie is None:
                continue
            counts[etype]["has_grounding"] += 1
            if top_curie in mention.gold_synonyms:
                counts[etype]["correct"] += 1

    return _counts_to_df(counts)


def evaluate_predictions(
    docs: list[DocumentExample],
    predictions: dict[str, dict[str, list]],
) -> pd.DataFrame:
    """Evaluate a model's predicted re-ranking of candidates by computing
    precision, recall, and F1, grouped by entity types found in the corpus.

    Params:
    -------
    docs :
        Held-out set of documents
    predictions :
        Dictionary that maps doc_id: {mention_text: list[ScoredMatch]}

    Returns:
    --------
    pd.DataFrame :
        Evaluation results grouped by entity type.
    """
    counts = defaultdict(lambda: {"correct": 0, "has_grounding": 0,
                                  "total": 0})

    for doc in docs:
        doc_preds = predictions.get(doc.doc_id, {})
        for mention in doc.mentions:
            etype = mention.entity_type
            counts[etype]["total"] += 1
            preds = doc_preds.get(mention.text, mention.candidates)
            if not preds:
                continue
            counts[etype]["has_grounding"] += 1
            top_curie = f"{preds[0].term.db}:{preds[0].term.id}"
            if top_curie in mention.gold_synonyms:
                counts[etype]["correct"] += 1

    return _counts_to_df(counts)


def filter_ambiguous_mentions(
    docs: list[DocumentExample],
    max_score_gap: float = 0.05,
    min_candidates: int = 2,
    min_top_score: float = 0.3,
) -> list[DocumentExample]:
    """Filter documents to only include mentions where Gilda's candidate
    list is genuinely ambiguous. A mention is considered ambiguous when it
    has at least `min_candidates` candidates, the gap between the 1st and 2nd
    candidate scores is at most `max_score_gap`, and the top candidate score
    is at least `min_top_score` (to exclude poor-quality matches).

    Params:
    ------
    docs :
        List of DocumentExample instances.
    max_score_gap :
        Maximum allowed difference between the top two candidate scores.
    min_candidates :
        Minimum number of candidates required.
    min_top_score :
        Minimum score for the top candidate.

    Returns:
    --------
    list[DocumentExample]
        Filtered documents containing only ambiguous mentions. Documents
        with no remaining mentions are dropped.
    """
    total_mentions = 0
    kept_mentions = 0
    filtered = []
    for doc in docs:
        amb_mentions = []
        for m in doc.mentions:
            total_mentions += 1
            if len(m.candidates) < min_candidates:
                continue
            top = m.candidates[0].score
            second = m.candidates[1].score
            gap = top - second
            if gap <= max_score_gap and top >= min_top_score:
                amb_mentions.append(m)
                kept_mentions += 1
        if amb_mentions:
            filtered.append(DocumentExample(
                doc_id=doc.doc_id,
                mentions=amb_mentions,
            ))
    print(f"Ambiguity filter: kept {kept_mentions}/{total_mentions} mentions "
          f"({kept_mentions/total_mentions:.1%}) across {len(filtered)} docs "
          f"(gap<={max_score_gap}, candidates>={min_candidates}, "
          f"top>={min_top_score})")
    return filtered


def evaluate_baseline(docs: list[DocumentExample]) -> pd.DataFrame:
    """"Evaluate Gilda's default pre-disambiguation candidate ranking.
    """
    counts = defaultdict(lambda: {"correct": 0, "has_grounding": 0, "total": 0})

    for doc in docs:
        for mention in doc.mentions:
            etype = mention.entity_type
            counts[etype]["total"] += 1
            if not mention.candidates:
                continue
            counts[etype]["has_grounding"] += 1
            top_curie = f"{mention.candidates[0].term.db}:{mention.candidates[0].term.id}"
            if top_curie in mention.gold_synonyms:
                counts[etype]["correct"] += 1

    return _counts_to_df(counts)


def _counts_to_df(counts: dict) -> pd.DataFrame:
    """Converts counts dictionary to a nice DataFrame.
    """
    rows = []
    for etype in sorted(counts.keys()):
        c = counts[etype]
        precision = c["correct"] / c["has_grounding"] if c["has_grounding"] \
            else 0
        recall = c["correct"] / c["total"] if c["total"] else 0
        f1 = (2 * precision * recall / (precision + recall)) if \
            (precision + recall) else 0
        rows.append({
            "Entity Type": etype,
            "Correct": c["correct"],
            "Has Grounding": c["has_grounding"],
            "Total": c["total"],
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
        })
    # Total row
    total = {k: sum(r[k] for r in rows) for k in ("Correct",
                                                  "Has Grounding", "Total")}
    p = total["Correct"] / total["Has Grounding"] if (
        total)["Has Grounding"] else 0
    r = total["Correct"] / total["Total"] if total["Total"] else 0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0
    rows.append({
        "Entity Type": "Total",
        **total,
        "Precision": round(p, 3),
        "Recall": round(r, 3),
        "F1": round(f1, 3),
    })
    return pd.DataFrame(rows)


def compare_methods(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Print a table comparing results across methods.
    """
    names = list(results.keys())
    base = results[names[0]][["Entity Type"]].copy()
    for name in names:
        df = results[name]
        base[f"{name}_Correct"] = df["Correct"].values
        base[f"{name}_F1"] = df["F1"].values

    base_col = f"{names[0]}_Correct"
    for name in names[1:]:
        base[f"{name}_Delta"] = base[f"{name}_Correct"] - base[base_col]
    return base

def comparison_table(docs, predictions):
    """Generates a comparison table with per entity_type Total, Gilda F1,
    Attn F1, Gains, Losses, and Net.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {"total": 0, "g_has": 0, "g_corr": 0,
                               "a_has": 0, "a_corr": 0, "gains": 0, "losses": 0})

    def curie(c):
        return f"{c.term.db}:{c.term.id}"

    for doc in docs:
        dp = predictions.get(doc.doc_id, {})
        for m in doc.mentions:
            r = agg[m.entity_type]
            r["total"] += 1
            base = m.candidates
            attn = dp.get(m.text, m.candidates)
            b_ok = bool(base) and curie(base[0]) in m.gold_synonyms
            a_ok = bool(attn) and curie(attn[0]) in m.gold_synonyms
            r["g_has"] += bool(base)
            r["a_has"] += bool(attn)
            r["g_corr"] += b_ok
            r["a_corr"] += a_ok
            r["gains"] += (a_ok and not b_ok)
            r["losses"] += (b_ok and not a_ok)

    def f1(corr, has, total):
        p = corr / has if has else 0
        rec = corr / total if total else 0
        return round(2 * p * rec / (p + rec), 3) if (p + rec) else 0

    def gl_ratio(gains, losses):
        if losses == 0:
            return float("inf") if gains else 0.0
        return round(gains / losses, 2)

    rows, tot = [], defaultdict(int)
    for et in sorted(agg):
        r = agg[et]
        for k in r:
            tot[k] += r[k]
        rows.append({
            "Entity Type": et, "Total": r["total"],
            "Gilda F1": f1(r["g_corr"], r["g_has"], r["total"]),
            "Attn F1": f1(r["a_corr"], r["a_has"], r["total"]),
            "Gains": r["gains"],
            "Losses": r["losses"],
            "Net": r["gains"] - r["losses"],
            "G/L": gl_ratio(r["gains"], r["losses"]),
        })
    rows.append({
        "Entity Type": "Total", "Total": tot["total"],
        "Gilda F1": f1(tot["g_corr"], tot["g_has"], tot["total"]),
        "Attn F1": f1(tot["a_corr"], tot["a_has"], tot["total"]),
        "Gains": tot["gains"],
        "Losses": tot["losses"],
        "Net": tot["gains"] - tot["losses"],
        "G/L": gl_ratio(tot["gains"], tot["losses"]),
    })
    import pandas as pd
    return pd.DataFrame(rows)


def filter_docs_by_source(docs, src):
    """Handles merged documents whose source is a merged string
    (e.g. 'bc5cdr+ncbi_disease').
    """
    out = []
    for d in docs:
        ms = [m for m in d.mentions if src in m.source_datasets]
        if ms:
            out.append(DocumentExample(doc_id=d.doc_id, mentions=ms,
                                       source=d.source, split=d.split))
    return out




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["feasibility", "evaluate",
                                           "compare"], required=True)
    parser.add_argument("--deepwalk-model", default=None,
                        help="Path to deepwalk_model_v1.pkl")
    parser.add_argument("--graph-path", default=None,
                        help="Path to multigraph_v1.pkl for PPR")
    parser.add_argument("--model-path", default=None,
                        help="Path to trained attention model")
    parser.add_argument("--embedding-cache", default="embedding_cache.pkl")
    parser.add_argument("--equivalences", default=None)
    parser.add_argument("--ambiguous-only", action="store_true",
                        help="Evaluate only on ambiguous mentions")
    parser.add_argument("--max-score-gap", type=float, default=0.05,
                        help="Max gap between top two candidate scores "
                             "(if --ambiguous-only)")
    parser.add_argument("--min-candidates", type=int, default=2,
                        help="Min number of candidates for a mention "
                             "(if --ambiguous-only)")
    parser.add_argument("--min-top-score", type=float, default=0.3,
                        help="Min score for the top candidate "
                             "(if --ambiguous-only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", nargs="+", default=["bioid"])
    parser.add_argument("--corpus-cache", default=None)
    args = parser.parse_args()

    if args.mode == "feasibility":
        embedder = CandidateEmbedder(device=args.device)
        feasibility_check(embedder, deepwalk_model_path=args.deepwalk_model)

    elif args.mode in ("evaluate", "compare"):
        import json
        from .data import load_corpus, make_splits
        from .train import precompute_embeddings, predict_document, load_model

        equivalences = {}
        if args.equivalences:
            with open(args.equivalences) as f:
                equivalences = json.load(f)

        docs = load_corpus(args.datasets, equivalences=equivalences,
                           merged_cache=args.corpus_cache)
        _, _, test_docs = make_splits(docs)

        if args.ambiguous_only:
            test_docs = filter_ambiguous_mentions(
                test_docs,
                max_score_gap=args.max_score_gap,
                min_candidates=args.min_candidates,
                min_top_score=args.min_top_score,
            )

        all_results = {}

        # Gilda baseline
        baseline_df = evaluate_baseline(test_docs)
        all_results["Gilda"] = baseline_df
        print("\n=== Gilda Baseline ===")
        print(baseline_df.to_markdown(index=False))

        # PPR baseline
        if args.graph_path:
            print("\n=== PPR Baseline ===")
            ppr_df = evaluate_ppr(test_docs, args.graph_path)
            all_results["PPR"] = ppr_df
            print(ppr_df.to_markdown(index=False))

        # DeepWalk baseline
        if args.deepwalk_model:
            print("\n=== DeepWalk Cosine Baseline ===")
            dw_df = evaluate_deepwalk_cosine(test_docs, args.deepwalk_model)
            all_results["DeepWalk"] = dw_df
            print(dw_df.to_markdown(index=False))

        # Attention-based model
        if args.model_path:
            embedder = CandidateEmbedder(device=args.device)
            cache = precompute_embeddings(test_docs, embedder,
                                          args.embedding_cache)
            model = load_model(args.model_path, device=args.device)

            predictions = {}
            for doc in test_docs:
                predictions[doc.doc_id] = predict_document(
                    doc, cache, model, device=args.device,
                )

            model_df = evaluate_predictions(test_docs, predictions)
            all_results["Attention"] = model_df
            print("\n=== Attention Model ===")
            print(model_df.to_markdown(index=False))

            print("\n=== Combined (all sources) ===")
            print(comparison_table(test_docs, predictions).to_markdown(index=False))

            print("\n=== Per source ===")
            srcs = sorted({s for d in test_docs for m in d.mentions
                           for s in m.source_datasets})
            for src in srcs:
                sub = filter_docs_by_source(test_docs, src)
                print(f"\n[{src}]")
                print(comparison_table(sub, predictions).to_markdown(index=False))

        # Comparison table
        if len(all_results) > 1 and args.mode == "compare":
            comp = compare_methods(all_results)
            print("\n=== Method Comparison ===")
            print(comp.to_markdown(index=False))
