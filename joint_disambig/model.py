"""Self-attention model for joint candidate scoring."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JointDisambiguator(nn.Module):
    """Disambiguates a mention's Gilda candidates by attending to the Gilda
    candidates of other mentions from the same source (document, experimental
    dataset, etc.). Each candidate is represented as a numerical vector of
    [LLM_embedding; gilda_score], which is projected into a lower-dimensional
    latent space, then passed through self-attention to let candidates from
    different mentions talk to each other as they are updated. A linear head
    then produces a scalar score per candidate, which is merged with the
    original Gilda score to produce the final grounding score.

     Params:
     -------
     embed_dim :
        Dimensionality of LLM embeddings (768 for PubMedBERT)
     hidden_dim :
        Dimensionality of hidden layers
     n_heads :
        Number of attention heads
     dropout :
        Dropout rate for training
    """
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim + 1, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: Tensor,
        gilda_scores: Tensor,
        mention_ids: Tensor,
        cross_mention_only: bool = False,
    ) -> Tensor:
        """Score all candidates jointly.

        Params:
        -------
        embeddings : (N, embed_dim)
            LLM embeddings for N candidates.
        gilda_scores : (N, 1)
            Gilda's lexical scores for N candidates.
        mention_ids : (N,)
            Integer ID for each candidate indicating its mention group. Used
            to group candidates for computing the loss.
        cross_mention_only : bool
            If True, use an attention mask to block attention between candidates
            for the same mention. Simple change to the attention pattern.

        Returns:
        --------
        scores : (N,)
        """
        x = torch.cat([embeddings, gilda_scores], dim=-1)
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)

        # Build an attention mask if cross-mention only
        attn_mask = None
        if cross_mention_only:
            # Block within-mention attn but allow self-attn
            N = mention_ids.shape[0]
            same_mention = mention_ids.unsqueeze(0) == mention_ids.unsqueeze(1)
            diag = torch.eye(N, dtype=torch.bool, device=mention_ids.device)
            attn_mask = same_mention & ~diag  # i.e., block siblings, allow self

        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x.squeeze(0)
        return self.score_head(x).squeeze(-1)


class GatedJointDisambiguator(nn.Module):
    """JointDisambiguator with a learned confidence gate. The gate looks at
    three statistics per mention––the number of candidates, the score gap
    between the top 2 candidates, and the score of the top candidate––and
    outputs a blending weight between 0 and 1, where 0 means defer to the
    original Gilda scores/ranking and 1 means trust the new model-learned
    scores.

    Concretely, Final score = gate * attention_score + (1 - gate) * gilda_score.

    The gate is per-mention, meaning all candidates of a mention share the
    same gate value. This forces the gate to dynamically determine when to
    trust the new model-learned scores.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_gate_features: int = 3,
    ):
        super().__init__()
        # Has the same attention architecture as JointDisambiguator
        self.input_proj = nn.Linear(embed_dim + 1, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # MLP for the gate: maps per-mention features to a scalar in [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(n_gate_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        embeddings: Tensor,
        gilda_scores: Tensor,
        mention_ids: Tensor,
        gate_features: Tensor,
        cross_mention_only: bool = False,
    ) -> Tensor:
        """Score all candidates jointly with a learned confidence gate.

        Params:
        -------
        embeddings : (N, embed_dim)
            LLM embeddings for N candidates.
        gilda_scores : (N, 1)
            Gilda's lexical scores for N candidates.
        mention_ids : (N,)
            Integer ID for each candidate indicating its mention group. Used
            to group candidates for computing the loss.
        gate_features : (M, n_gate_features)
            Per-mention features ([score_gap, n_candidates, top_score])
        cross_mention_only : bool
            If True, use an attention mask to block attention between candidates
            for the same mention. Simple change to the attention pattern.

        Returns:
        --------
        scores : (N,)
        """
        # Same as JointDisambiguator
        x = torch.cat([embeddings, gilda_scores], dim=-1)
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)

        attn_mask = None
        if cross_mention_only:
            N = mention_ids.shape[0]
            same_mention = mention_ids.unsqueeze(0) == mention_ids.unsqueeze(1)
            diag = torch.eye(N, dtype=torch.bool, device=mention_ids.device)
            attn_mask = same_mention & ~diag

        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x.squeeze(0)
        attn_scores = self.score_head(x).squeeze(-1)  # (N,)

        # New per-mention gate: broadcast (M, 1) to (N,)
        gate_values = self.gate(gate_features).squeeze(-1)  # (M,)
        per_candidate_gate = gate_values[mention_ids]  # (N,)

        # Blend model scores with original gilda scores
        return per_candidate_gate * attn_scores + (1 - per_candidate_gate) * gilda_scores.squeeze(-1)


def compute_loss(
    scores: Tensor,
    mention_ids: Tensor,
    gold_indices: Tensor,
) -> Tensor:
    """For computing per-mention cross-entropy loss and averaging over valid mentions.

    Params:
    -------
    scores : (N,)
        Raw model scores for all candidates.
    mention_ids : (N,)
        Specifies which mention each candidate belongs to.
    gold_indices : (M,)
        The index of the correct candidate within each mention's candidates. A
        value of -1 means the true grounding isn't in the candidate list and that
        mention's candidates are excluded from the loss.

    Returns:
    --------
    loss : scalar tensor
    """
    losses = []
    for m_id in range(gold_indices.shape[0]):
        gold_idx = gold_indices[m_id].item()
        if gold_idx < 0:
            continue
        mask = mention_ids == m_id
        mention_scores = scores[mask]
        if mention_scores.shape[0] == 0:
            continue
        log_probs = F.log_softmax(mention_scores, dim=0)
        losses.append(-log_probs[gold_idx])
    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)
    return torch.stack(losses).mean()
