"""Use a pretrained PubMedBERT model from HuggingFace to embed gilda candidates."""
import os
import pickle
from typing import Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)

# Human-readable namespace labels for embedding context
NAMESPACE_LABELS = {
    "HGNC": "gene", "UP": "protein", "FPLX": "protein family",
    "CHEBI": "chemical", "PUBCHEM": "chemical", "GO": "gene ontology term",
    "MESH": "biomedical concept", "DOID": "disease", "HP": "phenotype",
    "EFO": "experimental factor", "CL": "cell type", "BTO": "tissue",
    "NCIT": "NCI concept", "TAXONOMY": "organism", "IP": "InterPro domain",
    "PF": "Pfam domain",
}


def _build_embedding_text(term, grounder=None) -> str:
    """Build a rich text string for embedding a Gilda Term. For example, instead of embedding a
    candidate like HGNC:3467 by only its entry_name "ESR1", embed a string that includes this
    name plus a long-form name and a namespace label like "ESR1, estrogen receptor 1 (gene)",
    which takes the form: "{entry_name}, {long_form_name} ({namespace_label})"

    Params:
    -------
    term :
        Gilda Term
    grounder :
        Gilda Grounder instance (optional)

    Returns:
    --------
    str : rich text string to be embedded

    """
    parts = [term.entry_name]
    if grounder is not None:
        best_name = None
        for norm_text, terms in grounder.entries.items():
            for t in terms:
                if (t.db == term.db and t.id == term.id
                        and t.status == "name"
                        and t.text != term.entry_name):
                    if best_name is None or len(t.text) > len(best_name):
                        best_name = t.text
        if best_name:
            parts.append(best_name)

    ns_label = NAMESPACE_LABELS.get(term.db)
    if ns_label:
        return ", ".join(parts) + f" ({ns_label})"
    return ", ".join(parts)


class CandidateEmbedder:
    """Embeds Gilda candidate entities using a pretrained LLM.

    Params:
    -------
    model_name :
        HuggingFace model identifier. Default is PubMedBERT (now known as BiomedBERT).
    device :
        'cpu', 'mps', or 'cuda'.
    cache_path :
        If this is provided, persist the embedding cache to this pickle file.
    grounder :
        Optional Gilda Grounder instance for looking up full entity names
        from the term table. If provided, allows for embedding richer entity texts
        (e.g., "ESR1, estrogen receptor 1 (gene)" instead of "ESR1").
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        cache_path: Optional[str] = None,
        grounder=None,
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self.cache_path = cache_path
        self.grounder = grounder
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self._cache = pickle.load(f)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts. Returns (N, hidden_dim) array via [CLS] pooling.
        """
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tok = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(self.device)
            out = self.model(**tok)
            all_vecs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.concatenate(all_vecs, axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns 1-D array.
        """
        return self.embed_texts([text])[0]

    def embed_candidate(self, term) -> np.ndarray:
        """Embed a Gilda Term and cache its embedding by the key "(db, id)".

        If a grounder was provided at init, uses rich text (entry_name +
        full name + namespace label). Otherwise, uses entry_name only.
        """
        key = (term.db, term.id)
        if key not in self._cache:
            text = _build_embedding_text(term, self.grounder)
            self._cache[key] = self.embed_text(text)
        return self._cache[key]

    def embed_candidates(self, scored_matches: list) -> np.ndarray:
        """Embed all candidates in a ScoredMatch list. Returns (N, hidden_dim)
        array.
        """
        vecs = [self.embed_candidate(m.term) for m in scored_matches]
        return np.stack(vecs)

    def save_cache(self):
        """Save the embedding cache to disk
        """
        if self.cache_path:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size
