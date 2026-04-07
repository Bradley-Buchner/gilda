"""Joint disambiguation module for Gilda that uses PubMedBERT embeddings and self-attention.
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
    """Retrun a list of re-ranked Gilda candidates using joint disambiguation.

    Params:
    -------
    mention_candidates :
        Mapping of text mentions text to their Gilda ScoredMatch lists from gilda.ground(). For joint
        disambiguation, all mentions should come from the same source document/dataset.
    model_path :
        Path to a trained disambiguation model.
    device :
        Device to use for inference (either 'cpu' or 'gpu')

    Returns:
    --------
    dict[str, list[ScoredMatch]] :
        Object with the same structure as mention_candidates but with re-ranked candidates.
    """
    global _model, _embedder, _cache

    if model_path is None:
        raise ValueError("model_path is required (no bundled model yet)")

    if _model is None:
        from .train import load_model
        from .embedder import CandidateEmbedder
        _model = load_model(model_path, device=device)
        _embedder = CandidateEmbedder(device=device)

    # Build an example document and run inference on it
    from .data import DocumentExample, MentionExample
    from .train import predict_document

    mentions = [
        MentionExample(text=text, entity_type="", candidates=cands)
        for text, cands in mention_candidates.items()
    ]
    doc = DocumentExample(doc_id="_runtime", mentions=mentions)

    for m in mentions:
        for cand in m.candidates:
            key = (cand.term.db, cand.term.id)
            if key not in _cache:
                _cache[key] = _embedder.embed_candidate(cand.term)

    return predict_document(doc, _cache, _model, device=device)
