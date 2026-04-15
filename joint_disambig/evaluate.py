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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.mode == "feasibility":
        embedder = CandidateEmbedder(device=args.device)
        feasibility_check(embedder, deepwalk_model_path=args.deepwalk_model)

    elif args.mode in ("evaluate", "compare"):
        import json
        from .data import load_bioid_corpus, split_by_document
        from .train import precompute_embeddings, predict_document, load_model

        equivalences = {}
        if args.equivalences:
            with open(args.equivalences) as f:
                equivalences = json.load(f)

        docs = load_bioid_corpus(equivalences=equivalences)
        _, _, test_docs = split_by_document(docs)

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

        # Comparison table
        if len(all_results) > 1 and args.mode == "compare":
            comp = compare_methods(all_results)
            print("\n=== Method Comparison ===")
            print(comp.to_markdown(index=False))
