"""Training loop, embedding precomputation, and inference for joint
disambiguation"""
import copy
import os
import pickle
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm

from .data import DocumentExample, MentionExample
from .embedder import CandidateEmbedder, _build_embedding_text
from .model import JointDisambiguator, GatedJointDisambiguator, compute_loss


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


def collate_document(
    doc: DocumentExample,
    embedding_cache: dict[tuple[str, str], np.ndarray],
    embed_dim: int = 768,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert a DocumentExample object into a tuple of model-ready tensors.
    Returns the tuple (embeddings, gilda_scores, mention_ids, gold_indices) or
    None if the document has no usable candidates.
    """
    all_embs, all_scores, all_mids = [], [], []
    gold_indices = []
    m_id = 0

    # Loop through a doc's mentions and candidates
    for mention in doc.mentions:
        if not mention.candidates:
            continue
        mention_embs = []
        for cand in mention.candidates:
            key = (cand.term.db, cand.term.id)
            emb = embedding_cache.get(key)
            if emb is None:
                emb = np.zeros(embed_dim)
            mention_embs.append(emb)
            all_scores.append(cand.score)
            all_mids.append(m_id)
        all_embs.extend(mention_embs)
        gold_indices.append(mention.gold_index if mention.gold_index is not
                                                  None else -1)
        m_id += 1

    if not all_embs:
        return None

    embeddings = torch.tensor(np.stack(all_embs), dtype=torch.float32)
    gilda_scores = torch.tensor(all_scores, dtype=torch.float32).unsqueeze(-1)
    mention_ids = torch.tensor(all_mids, dtype=torch.long)
    gold_idx = torch.tensor(gold_indices, dtype=torch.long)

    return embeddings, gilda_scores, mention_ids, gold_idx


def compute_gate_features(doc: DocumentExample) -> Optional[torch.Tensor]:
    """Compute per-mention features used by the gate MLP (score_gap,
    n_candidates, and top_score). Returns a tensor with size (M, 3) where
    M = number of mentions that have candidates, or None if no mentions have
    candidates.
    """
    features = []
    for mention in doc.mentions:
        if not mention.candidates:
            continue
        top_score = mention.candidates[0].score
        if len(mention.candidates) >= 2:
            score_gap = top_score - mention.candidates[1].score
        else:
            score_gap = top_score
        n_cands = len(mention.candidates)
        features.append([score_gap, n_cands, top_score])
    if not features:
        return None
    return torch.tensor(features, dtype=torch.float32)


def train(
    train_docs: list[DocumentExample],
    val_docs: list[DocumentExample],
    embedding_cache: dict,
    model: JointDisambiguator,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
    cross_mention_only: bool = False,
) -> JointDisambiguator:
    """Train a model with early stopping on validation loss and return the
    best model checkpoint.
    """
    device = torch.device(device)
    model = model.to(device)
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
            batch = collate_document(doc, embedding_cache)
            if batch is None:
                continue
            embs, gs, mids, golds = [t.to(device) for t in batch]
            scores = model(embs, gs, mids,
                           cross_mention_only=cross_mention_only)
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
                batch = collate_document(doc, embedding_cache)
                if batch is None:
                    continue
                embs, gs, mids, golds = [t.to(device) for t in batch]
                scores = model(embs, gs, mids,
                               cross_mention_only=cross_mention_only)
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


def predict_document(
    doc: DocumentExample,
    embedding_cache: dict,
    model: JointDisambiguator,
    device: str = "cpu",
    cross_mention_only: bool = False,
) -> dict[str, list]:
    """Run inference on a single document and return {mention_text: re-ranked
    ScoredMatch list}.
    """
    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()

    batch = collate_document(doc, embedding_cache)
    results = {}
    if batch is None:
        for m in doc.mentions:
            results[m.text] = m.candidates
        return results

    embs, gs, mids, _ = [t.to(device_t) for t in batch]
    with torch.no_grad():
        scores = model(embs, gs, mids,
                       cross_mention_only=cross_mention_only).cpu().numpy()

    # Map scores back to mentions and re-rank candidates
    m_id = 0
    score_idx = 0
    for mention in doc.mentions:
        if not mention.candidates:
            results[mention.text] = mention.candidates
            continue
        n_cands = len(mention.candidates)
        mention_scores = scores[score_idx : score_idx + n_cands]
        ranked_indices = np.argsort(-mention_scores)
        results[mention.text] = [mention.candidates[i] for i in ranked_indices]
        score_idx += n_cands
        m_id += 1

    return results


def save_model(model: JointDisambiguator, path: str):
    """Save model checkpoint to disk.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "embed_dim": model.input_proj.in_features - 1,
            "hidden_dim": model.input_proj.out_features,
            "n_heads": model.attention.num_heads,
        },
    }, path)


def load_model(path: str, device: str = "cpu") -> JointDisambiguator:
    """Load a saved model checkpoint from disk.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = JointDisambiguator(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train joint disambiguation model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="joint_disambig_model.pt")
    parser.add_argument("--embedding-cache", default="embedding_cache.pkl")
    parser.add_argument("--equivalences", default=None, help="Path to "
                                                             "equivalences.json")
    args = parser.parse_args()

    from .data import load_bioid_corpus, split_by_document, report_statistics
    import json

    equivalences = {}
    if args.equivalences and os.path.exists(args.equivalences):
        with open(args.equivalences) as f:
            equivalences = json.load(f)

    docs = load_bioid_corpus(equivalences=equivalences)
    report_statistics(docs)
    train_docs, val_docs, test_docs = split_by_document(docs)
    print(f"Split: {len(train_docs)} train, {len(val_docs)} val, "
          f"{len(test_docs)} test")

    embedder = CandidateEmbedder(device=args.device,
                                 cache_path=args.embedding_cache)
    cache = precompute_embeddings(train_docs + val_docs + test_docs, embedder,
                                  args.embedding_cache)

    model = JointDisambiguator(embed_dim=embedder.embed_dim)
    model = train(
        train_docs, val_docs, cache, model,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=args.device,
    )
    save_model(model, args.output)
    print(f"Model saved to {args.output}")
