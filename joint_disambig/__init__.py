"""Joint disambiguation module for Gilda that uses PubMedBERT embeddings and self-attention.

The default model is a GatedJointDisambiguator with rich embeddings;
candidates are embedded using their full name + namespace label via frozen
PubMedBERT, and a learned confidence gate blends the attention signal with
Gilda's original lexical scores.
"""
from typing import Optional

_model = None
_embedder = None
_cache = {}
_jr = None


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
    global _model, _embedder

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

    # Re-rank with a shared JointReranker
    from .rerank import JointReranker
    global _jr
    if _jr is None:
        _jr = JointReranker(_model, grounder=None, device=device, cache=_cache, embedder=_embedder)

    texts = list(mention_candidates.keys())
    ranked = _jr.rerank([mention_candidates[t] for t in texts])
    return dict(zip(texts, ranked))