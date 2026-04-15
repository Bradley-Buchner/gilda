"""Joint disambiguation module for Gilda that uses PubMedBERT embeddings and self-attention.

The default model is a GatedJointDisambiguator with rich embeddings:
candidates are embedded using their full name + namespace label via frozen
PubMedBERT, and a learned confidence gate blends the attention signal with
Gilda's original lexical scores.
"""
from typing import Optional

_model = None
_embedder = None
_cache = {}


def disambiguate(
        mention_candidates: dict[str, list],
        model_path: Optional[str] = None,
        device: str = "cpu",
) -> dict[str, list]:
    """Return re-ranked Gilda candidates using joint disambiguation.

    Params:
    -------
    mention_candidates :
        Mapping of text mentions to their Gilda ScoredMatch lists from
        gilda.ground(). For joint disambiguation, all mentions should
        come from the same source document/dataset.
    model_path :
        Path to a trained disambiguation model checkpoint.
    device :
        Device to use for inference ('cpu' or 'cuda').

    Returns:
    --------
    dict[str, list[ScoredMatch]]
        Same structure as mention_candidates but with re-ranked candidates.
    """
    global _model, _embedder, _cache

    if model_path is None:
        raise ValueError("model_path is required (no bundled model yet)")

    if _model is None:
        import torch
        from .model import GatedJointDisambiguator
        from .embedder import CandidateEmbedder

        # Load checkpoint (supports both gated and plain models)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model_type = ckpt.get("model_type", "JointDisambiguator")

        if model_type == "GatedJointDisambiguator":
            _model = GatedJointDisambiguator(**ckpt["config"])
        else:
            from .model import JointDisambiguator
            _model = JointDisambiguator(**ckpt["config"])

        _model.load_state_dict(ckpt["state_dict"])
        _model.eval()

        # Use rich embeddings if the checkpoint was trained with them
        embedding_mode = ckpt.get("embedding_mode", "plain")
        if embedding_mode == "rich":
            from gilda.grounder import Grounder
            _embedder = CandidateEmbedder(device=device, grounder=Grounder())
        else:
            _embedder = CandidateEmbedder(device=device)

    # Build a temporary document and run inference
    from .data import DocumentExample, MentionExample
    from .train import collate_document, compute_gate_features
    import torch
    import numpy as np

    mentions = [
        MentionExample(text=text, entity_type="", candidates=cands)
        for text, cands in mention_candidates.items()
    ]
    doc = DocumentExample(doc_id="_runtime", mentions=mentions)

    # Embed any new candidates
    for m in mentions:
        for cand in m.candidates:
            key = (cand.term.db, cand.term.id)
            if key not in _cache:
                _cache[key] = _embedder.embed_candidate(cand.term)

    # Collate and run
    batch = collate_document(doc, _cache)
    if batch is None:
        return {m.text: m.candidates for m in mentions}

    device_t = torch.device(device)
    embs, gs, mids, _ = [t.to(device_t) for t in batch]

    with torch.no_grad():
        if isinstance(_model, GatedJointDisambiguator):
            gate_feats = compute_gate_features(doc)
            if gate_feats is None:
                return {m.text: m.candidates for m in mentions}
            scores = _model(embs, gs, mids, gate_feats.to(device_t))
        else:
            scores = _model(embs, gs, mids)
        scores = scores.cpu().numpy()

    # Re-rank
    results = {}
    score_idx = 0
    for mention in doc.mentions:
        if not mention.candidates:
            results[mention.text] = mention.candidates
            continue
        n_cands = len(mention.candidates)
        mention_scores = scores[score_idx: score_idx + n_cands]
        ranked = np.argsort(-mention_scores)
        results[mention.text] = [mention.candidates[i] for i in ranked]
        score_idx += n_cands

    return results
