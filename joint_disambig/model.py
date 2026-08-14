"""Self-attention model for joint candidate scoring."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, attn_mask):
        a, _ = self.attn(seq, seq, seq, attn_mask=attn_mask, need_weights=False)
        seq = self.norm1(seq + self.dropout(a))
        return self.norm2(seq + self.ff(seq))


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
        embeddings: torch.Tensor,
        gilda_scores: torch.Tensor,
        mention_ids: torch.Tensor,
        cross_mention_only: bool = False,
    ) -> torch.Tensor:
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
        embeddings: torch.Tensor,
        gilda_scores: torch.Tensor,
        mention_ids: torch.Tensor,
        gate_features: torch.Tensor,
        cross_mention_only: bool = False,
    ) -> torch.Tensor:
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


class GatedGildaBiasDisambiguator(GatedJointDisambiguator):
    """GatedJointDisambiguator plus a learnable Gilda-score-derived attention bias.
    """
    def __init__(self,
                 embed_dim: int = 768,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 n_gate_features: int = 3
                 ):
        super().__init__(embed_dim, hidden_dim, n_heads, dropout, n_gate_features)
        self.gilda_attn_bias = nn.Linear(1, 1)  # scalar gilda score -> scalar bias

    def forward(self, embeddings, gilda_scores, mention_ids,
                gate_features, cross_mention_only=False):
        x = torch.cat([embeddings, gilda_scores], dim=-1)
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)

        N = mention_ids.shape[0]
        key_bias = self.gilda_attn_bias(gilda_scores).squeeze(-1)
        attn_mask = key_bias.unsqueeze(0).expand(N, N).clone()

        if cross_mention_only:
            same_mention = mention_ids.unsqueeze(0) == mention_ids.unsqueeze(1)
            diag = torch.eye(N, dtype=torch.bool, device=mention_ids.device)
            attn_mask = attn_mask.masked_fill(same_mention & ~diag, float("-inf"))

        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x.squeeze(0)
        attn_scores = self.score_head(x).squeeze(-1)

        gate_values = self.gate(gate_features).squeeze(-1)
        per_candidate_gate = gate_values[mention_ids]
        return per_candidate_gate * attn_scores + (1 - per_candidate_gate) * gilda_scores.squeeze(-1)


class GatedGateOnlyDisambiguator(GatedJointDisambiguator):
    """Ablation that tests removing the Gilda score from the candidate feature
    vector and only using it for the confidence gate.
    """
    def __init__(self,
                 embed_dim: int = 768,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 n_gate_features: int = 3
                 ):
        super().__init__(embed_dim, hidden_dim, n_heads, dropout, n_gate_features)
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, embeddings, gilda_scores, mention_ids,
                gate_features, cross_mention_only=False):
        x = F.relu(self.input_proj(embeddings))
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
        attn_scores = self.score_head(x).squeeze(-1)

        gate_values = self.gate(gate_features).squeeze(-1)
        per_candidate_gate = gate_values[mention_ids]
        return per_candidate_gate * attn_scores + (1 - per_candidate_gate) * gilda_scores.squeeze(-1)


class GatedGFeatDisambiguator(GatedGateOnlyDisambiguator):
    """Concatenate the match properties of each candidate onto its feature vector
    instead of its gilda score.
    """
    wants_candidate_features = True

    def __init__(self,
                 embed_dim: int = 768,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 n_gate_features: int = 3,
                 n_cand_features: int = 10
                 ):
        super().__init__(embed_dim, hidden_dim, n_heads, dropout, n_gate_features)
        self.n_cand_features = n_cand_features
        self.input_proj = nn.Linear(embed_dim + n_cand_features, hidden_dim)

class _GFeatKeyBiasBase(GatedGFeatDisambiguator):
    """Base class for GatedGFeatDisambiguator that sets two args:
      _bias_cols: feature vector column indices feeding the bias (None = all 10)
      _per_head : if True, instantiates separate biases per attention head
                    (Linear -> n_heads), else uses one shared bias (Linear -> 1).
    """
    _bias_cols = None
    _per_head = False

    def __init__(self,
                 embed_dim: int = 768,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 n_gate_features: int = 3,
                 n_cand_features: int = 10
                 ):
        super().__init__(embed_dim, hidden_dim, n_heads, dropout, n_gate_features,
                         n_cand_features)
        in_dim = n_cand_features if self._bias_cols is None else len(self._bias_cols)
        self.feat_attn_bias = nn.Linear(in_dim, n_heads if self._per_head else 1)

    def forward(self, embeddings, gilda_scores, mention_ids,
                gate_features, cross_mention_only=False):
        feats = embeddings[:, self.embed_dim:]
        feats_b = feats if self._bias_cols is None else feats[:, self._bias_cols]
        x = F.relu(self.input_proj(embeddings))
        x = self.dropout(x)

        N = mention_ids.shape[0]
        bias = self.feat_attn_bias(feats_b)
        if self._per_head:
            H = self.attention.num_heads
            attn_mask = bias.t().unsqueeze(1).expand(H, N, N).clone()
        else:
            attn_mask = bias.squeeze(-1).unsqueeze(0).expand(N, N).clone()

        if cross_mention_only:
            same = mention_ids.unsqueeze(0) == mention_ids.unsqueeze(1)
            block = same & ~torch.eye(N, dtype=torch.bool, device=mention_ids.device)
            block = block.unsqueeze(0) if self._per_head else block
            attn_mask = attn_mask.masked_fill(block, float("-inf"))

        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x.squeeze(0)
        attn_scores = self.score_head(x).squeeze(-1)
        gate_values = self.gate(gate_features).squeeze(-1)
        per_candidate_gate = gate_values[mention_ids]
        return per_candidate_gate * attn_scores + (1 - per_candidate_gate) * gilda_scores.squeeze(-1)



class GatedGFeatStatusExactBiasDisambiguator(_GFeatKeyBiasBase):
    """GatedGFeatDisambiguator plus a single bias from the 'status' and 'exact' cols.
    """
    _bias_cols = [0, 1, 2, 3, 6]
    _per_head = False


class GatedGFeatStatusExactCtxDisambiguator(GatedGFeatStatusExactBiasDisambiguator):
    """GatedGFeatDisambiguator plus one unscored document-level context token per
    document. The context (CTX) token in the frozen PubMedBERT [CLS] embedding
    of a comma-separated string of all of a document's surface mentions.
    """
    wants_context = True

    def __init__(self,
                 embed_dim=768,
                 hidden_dim=128,
                 n_heads=4,
                 dropout=0.1,
                 n_gate_features=3,
                 n_cand_features=10,
                 context_dim=768
                 ):
        super().__init__(embed_dim, hidden_dim, n_heads, dropout,
                         n_gate_features, n_cand_features)
        self.context_dim = context_dim
        self.ctx_proj = nn.Linear(context_dim, hidden_dim)

    def forward(self, embeddings, gilda_scores, mention_ids, gate_features,
                cross_mention_only=False, context_emb=None):
        feats_b = embeddings[:, self.embed_dim:][:, self._bias_cols]
        x = self.dropout(F.relu(self.input_proj(embeddings)))
        N = mention_ids.shape[0]
        key_bias = self.feat_attn_bias(feats_b).squeeze(-1)

        if context_emb is not None:
            ctx = F.relu(self.ctx_proj(context_emb.view(1, -1)))
            seq = torch.cat([ctx, x], dim=0).unsqueeze(0)
            full_bias = torch.cat(
                [torch.zeros(1, device=key_bias.device), key_bias])
            attn_mask = full_bias.unsqueeze(0).expand(1 + N, 1 + N).clone()
            out, _ = self.attention(seq, seq, seq, attn_mask=attn_mask)
            x = out.squeeze(0)[1:]
        else:
            attn_mask = key_bias.unsqueeze(0).expand(N, N).clone()
            xo = x.unsqueeze(0)
            x = self.attention(xo, xo, xo, attn_mask=attn_mask)[0].squeeze(0)

        attn_scores = self.score_head(x).squeeze(-1)
        gate_values = self.gate(gate_features).squeeze(-1)
        per_candidate_gate = gate_values[mention_ids]
        return per_candidate_gate * attn_scores + (1 - per_candidate_gate) * gilda_scores.squeeze(-1)


def compute_loss(
    scores: torch.Tensor,
    mention_ids: torch.Tensor,
    gold_indices: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Computes per-mention cross-entropy loss and averages over valid mentions.

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
    temperature : float
        Denominator that divides model scores before the softmax to sharpen the
        distribution (when < 1), which restores gradient magnitude if the scores have
        a narrow range. Ranking within a mention is invariant to this, so evaluation is
        unaffected and only the training gradients change. Default is 1.0.

    Returns:
    --------
    loss : scalar torch.Tensor
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
        log_probs = F.log_softmax(mention_scores / temperature, dim=0)
        losses.append(-log_probs[gold_idx])
    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)
    return torch.stack(losses).mean()
