"""Training loop, embedding precomputation, and inference for joint
disambiguation"""
import copy
import os
import pickle
from typing import Optional
import numpy as np
import torch

from .data import DocumentExample
from .embedder import CandidateEmbedder, _build_embedding_text
from .model import (JointDisambiguator, GatedJointDisambiguator,
                    GatedGildaBiasDisambiguator, GatedGateOnlyDisambiguator,
                    GatedGFeatDisambiguator, GatedGFeatStatusExactBiasDisambiguator,
                    GatedGFeatStatusExactCtxDisambiguator, compute_loss)
from .rerank import JointReranker
from gilda import Grounder


def precompute_embeddings(
    docs: list[DocumentExample],
    embedder: CandidateEmbedder,
    cache_path: Optional[str] = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Embed all unique candidates across all documents in a corpus and cache
    to disk. Returns a dict that maps (db, id) keys to 1d embedding vectors.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded {len(cache)} cached embeddings from {cache_path}")
        return cache

    # Build the dict of unique candidate keys and embedding text values
    unique = {}
    for doc in docs:
        for m in doc.mentions:
            for cand in m.candidates:
                key = (cand.term.db, cand.term.id)
                if key not in unique:
                    unique[key] = _build_embedding_text(cand.term,
                                                        embedder.grounder)

    # Get embeddings for each candidate key using batches
    keys = list(unique.keys())
    texts = [unique[k] for k in keys]
    print(f"Embedding {len(texts)} unique candidates...")
    vectors = embedder.embed_texts(texts, batch_size=64)
    cache = {k: vectors[i] for i, k in enumerate(keys)}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Saved embedding cache to {cache_path}")
    return cache


def precompute_context_embeddings(
    docs: list[DocumentExample],
    embedder: CandidateEmbedder,
    cache_path: Optional[str] = None,
    max_length: int = 512,
) -> dict[str, np.ndarray]:
    """Embed one context string per document. Used by models that have
    wants_context=True.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded {len(cache)} cached context embeddings from {cache_path}")
        return cache

    ids, strings = [], []
    for doc in docs:
        seen = list(dict.fromkeys(m.text for m in doc.mentions))
        ids.append(doc.doc_id)
        strings.append(", ".join(seen))
    print(f"Embedding {len(strings)} document-context strings...")
    vectors = embedder.embed_texts(strings, batch_size=32, max_length=max_length)
    cache = {doc_id: vectors[i] for i, doc_id in enumerate(ids)}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Saved context cache to {cache_path}")
    return cache


def _context_tensor(context_cache, doc_id, device):
    """Look up a doc's context vector and tensorize it.
    """
    if not context_cache:
        return None
    v = context_cache.get(doc_id)
    return torch.tensor(v, dtype=torch.float32, device=device) if v is not None else None


def train(
    train_docs: list[DocumentExample],
    val_docs: list[DocumentExample],
    embedding_cache: dict,
    model: JointDisambiguator,
    grounder: Grounder,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
    cross_mention_only: bool = False,
    temperature: float = 1.0,
    context_cache: Optional[dict] = None,
) -> JointDisambiguator:
    """Train a model with early stopping on validation loss and return the
    best model checkpoint. 'context_cache' feeds the document-context token
    for models with wants_context=True and is ignored otherwise.
    """
    device = torch.device(device)
    jr = JointReranker(model, grounder, device=device, cache=embedding_cache)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train for one epoch
        model.train()
        train_loss = 0.0
        n_train = 0
        for doc in train_docs:
            tensors = jr.build_model_tensors([m.candidates for m in doc.mentions])
            if tensors is None:
                continue
            embs, gs, mids, gate_feats = tensors
            golds = torch.tensor(
                [m.gold_index if m.gold_index is not None else -1
                 for m in doc.mentions if m.candidates],
                dtype=torch.long, device=device)
            ctx = _context_tensor(context_cache, doc.doc_id, device)
            scores = jr.model(embs, gs, mids, gate_feats,
                              cross_mention_only=cross_mention_only,
                              context_emb=ctx)
            loss = compute_loss(scores, mids, golds)
            if loss.item() == 0.0:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train += 1

        # Validate after training for one epoch
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for doc in val_docs:
                tensors = jr.build_model_tensors([m.candidates for m in doc.mentions])
                if tensors is None:
                    continue
                embs, gs, mids, gate_feats = tensors
                golds = torch.tensor(
                    [m.gold_index if m.gold_index is not None else -1
                     for m in doc.mentions if m.candidates],
                    dtype=torch.long, device=device
                )
                ctx = _context_tensor(context_cache, doc.doc_id, device)
                scores = jr.model(embs, gs, mids, gate_feats,
                                  cross_mention_only=cross_mention_only,
                                  context_emb=ctx)
                loss = compute_loss(scores, mids, golds)
                val_loss += loss.item()
                n_val += 1

        avg_train = train_loss / max(n_train, 1)
        avg_val = val_loss / max(n_val, 1)
        print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_train:.4f}  "
              f"val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


_rerankers = {}

def _get_reranker(model, embedding_cache, device):
    """Build once and reuse a JointReranker per (model, device) so the cache doesn't
    get duplicated and PubMedBERT is only loaded once.
    """
    key = (id(model), str(device))
    jr = _rerankers.get(key)
    if jr is None:
        jr = JointReranker(model, grounder=None, device=device,
                           cache=embedding_cache)
        _rerankers[key] = jr
    return jr


def predict_document(
        doc: DocumentExample,
        embedding_cache: dict,
        model: JointDisambiguator,
        device: str = "cpu",
        context_cache: Optional[dict] = None,
) -> dict[str, list]:
    """Run inference on a single document and return {mention_text: re-ranked
    ScoredMatch list}. 'context_cache' feeds the document-context token
    for models with wants_context=True and is ignored otherwise.
    """
    jr = _get_reranker(model, embedding_cache, device)
    with_cands = [m for m in doc.mentions if m.candidates]
    ctx = _context_tensor(context_cache, doc.doc_id, device) \
        if getattr(model, "wants_context", False) else None
    ranked = jr.rerank([m.candidates for m in with_cands],
                       context_emb=ctx) if with_cands else []
    results = {m.text: r for m, r in zip(with_cands, ranked)}
    for m in doc.mentions:
        results.setdefault(m.text, m.candidates)
    return results


def save_model(model: JointDisambiguator, path: str,
               embedding_mode: str = "plain"):
    """Save model checkpoint to disk with model type and embedding mode
    metadata.
    """
    model_type = type(model).__name__
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_type": model_type,
        "config": {
            "embed_dim": getattr(model, "embed_dim", model.input_proj.in_features - 1),
            "hidden_dim": model.input_proj.out_features,
            "n_heads": model.attention.num_heads,
            **({"n_cand_features": model.n_cand_features}
               if hasattr(model, "n_cand_features") else {}),
            **({"context_dim": model.context_dim}
               if hasattr(model, "context_dim") else {}),
        },
        "embedding_mode": embedding_mode,
    }, path)


def load_model(path: str, device: str = "cpu") -> JointDisambiguator:
    """Load a saved model checkpoint from disk.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model_type = ckpt.get("model_type", "JointDisambiguator")
    if model_type == "GatedGFeatStatusExactCtxDisambiguator":
        model = GatedGFeatStatusExactCtxDisambiguator(**ckpt["config"])
    elif model_type == "GatedGFeatStatusExactBiasDisambiguator":
        model = GatedGFeatStatusExactBiasDisambiguator(**ckpt["config"])
    elif model_type == "GatedGFeatDisambiguator":
        model = GatedGFeatDisambiguator(**ckpt["config"])
    elif model_type == "GatedGateOnlyDisambiguator":
        model = GatedGateOnlyDisambiguator(**ckpt["config"])
    elif model_type == "GatedGildaBiasDisambiguator":
        model = GatedGildaBiasDisambiguator(**ckpt["config"])
    elif model_type == "GatedJointDisambiguator":
        model = GatedJointDisambiguator(**ckpt["config"])
    else:
        model = JointDisambiguator(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model



if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Train joint disambiguation model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="joint_disambig/model_checkpoint.pt")
    parser.add_argument("--embedding-cache",
                        default="joint_disambig/embedding_cache_rich.pkl")
    parser.add_argument("--context-cache", default=None,
                        help="path to/for the per-document context embedding "
                             "cache (required for gated_gfeat_statusexactctx)")
    parser.add_argument("--equivalences", default=None,
                        help="Path to equivalences.json")
    parser.add_argument("--model-type",
                        choices=["plain", "gated", "gated_gbias", "gated_gateonly",
                                 "gated_gfeat", "gated_gfeat_statusexactbias",
                                 "gated_gfeat_statusexactctx"],
                        default="gated",
                        help="Model variant")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature param for sharpening the logit"
                             "distribution")
    parser.add_argument("--datasets", nargs="+", default=["bioid"],
                        help="Sources to combine, e.g. 'bioid bc5cdr nlmchem "
                             "ncbi_disease'. Default is 'bioid' (original pipeline).")
    parser.add_argument("--corpus-cache", default=None,
                        help="Path to cache or load the merged corpus pickle "
                             "(skips re-grounding on re-runs).")
    args = parser.parse_args()

    from .data import load_corpus, make_splits, report_statistics

    equivalences = {}
    if args.equivalences and os.path.exists(args.equivalences):
        with open(args.equivalences) as f:
            equivalences = json.load(f)

    grounder = Grounder()
    docs = load_corpus(args.datasets, grounder=grounder,
                       equivalences=equivalences,
                       merged_cache=args.corpus_cache)
    report_statistics(docs)
    train_docs, val_docs, test_docs = make_splits(docs)
    print(f"Split: {len(train_docs)} train, {len(val_docs)} val, "
          f"{len(test_docs)} test")

    # Rich embeddings (pass grounder for full names + species labels)
    embedder = CandidateEmbedder(device=args.device, grounder=grounder)
    cache = precompute_embeddings(train_docs + val_docs + test_docs,
                                  embedder, args.embedding_cache)

    if args.model_type == "gated_gfeat_statusexactctx":
        model = GatedGFeatStatusExactCtxDisambiguator(embed_dim=embedder.embed_dim)
    elif args.model_type == "gated_gfeat_statusexactbias":
        model = GatedGFeatStatusExactBiasDisambiguator(embed_dim=embedder.embed_dim)
    elif args.model_type == "gated_gfeat":
        model = GatedGFeatDisambiguator(embed_dim=embedder.embed_dim)
    elif args.model_type == "gated_gateonly":
        model = GatedGateOnlyDisambiguator(embed_dim=embedder.embed_dim)
    elif args.model_type == "gated_gbias":
        model = GatedGildaBiasDisambiguator(embed_dim=embedder.embed_dim)
    elif args.model_type == "gated":
        model = GatedJointDisambiguator(embed_dim=embedder.embed_dim)
    else:
        model = JointDisambiguator(embed_dim=embedder.embed_dim)

    ctx_cache = None
    if getattr(model, "wants_context", False):
        ctx_cache = precompute_context_embeddings(
            train_docs + val_docs + test_docs, embedder, args.context_cache)

    model = train(
        train_docs, val_docs, cache, model, grounder,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=args.device, temperature=args.temperature,
        context_cache=ctx_cache,
    )
    save_model(model, args.output, embedding_mode="rich")
    print(f"Model saved to {args.output}")
