"""BioCreative Bio-ID corpus loading and preprocessing for joint disambiguation.

Imports utilities for ID normalization, entity type classification, synonym expansion, and organism priority from the
BioIDBenchmarker class and module-level helpers in gilda/benchmarks/bioid_evaluation.py.
"""
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import pystow
from tqdm import tqdm

from gilda.grounder import Grounder, ScoredMatch

# Just an import of shared utilities from the benchmarks module
_BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "benchmarks",
)
if _BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARKS_DIR)
from bioid_evaluation import (
    BioIDBenchmarker,
    MODULE,
    URL,
)

_normalize_id = BioIDBenchmarker._normalize_id
_normalize_ids = BioIDBenchmarker._normalize_ids
_get_entity_type = BioIDBenchmarker._get_entity_type



@dataclass
class MentionExample:
    """For a single mention's candidates and ground truth label
    """
    text: str
    entity_type: str
    candidates: list
    gold_curies: set = field(default_factory=set)
    gold_synonyms: set = field(default_factory=set)
    gold_index: Optional[int] = None


@dataclass
class DocumentExample:
    """For all mentions from one document, forming one training example
    """
    doc_id: str
    mentions: list = field(default_factory=list)



# for _get_benchmarker
_benchmarker: Optional[BioIDBenchmarker] = None

def _get_benchmarker(
    grounder: Optional[Grounder] = None,
    equivalences: Optional[dict] = None,
) -> BioIDBenchmarker:
    """For lazily creating a BioIDBenchmarker to do synonym expansion and organism priority
    """
    global _benchmarker
    if _benchmarker is None:
        _benchmarker = BioIDBenchmarker(
            grounder=grounder or Grounder(),
            equivalences=equivalences or {},
        )
    return _benchmarker


def load_bioid_corpus(
    grounder: Optional[Grounder] = None,
    equivalences: Optional[dict] = None,
) -> list[DocumentExample]:
    """Load the BioCreative BioID corpus as a DocumentExample list.

    Params:
    ------
    grounder :
        Gilda Grounder instance. If None, uses default.
    equivalences :
        Equivalence mappings dict (e.g., from equivalences.json).
        
    Returns:
    --------
    list[DocumentExample]
        A list containing DocumentExample objects, each representing a document 
        with its mentions and associated disambiguation information.
    """

    if grounder is None:
        grounder = Grounder()
    benchmarker = _get_benchmarker(grounder, equivalences)

    print("Loading BioID annotations...")
    df = MODULE.ensure_tar_df(
        url=URL,
        inner_path="BioIDtraining_2/annotations.csv",
        read_csv_kwargs=dict(sep=",", low_memory=False),
    )
    df["obj"] = df["obj"].apply(_normalize_ids)
    df["obj_synonyms"] = df["obj"].apply(benchmarker.get_synonym_set)
    df["entity_type"] = df.apply(
        lambda row: _classify_entity_type(row.obj, row.obj_synonyms), axis=1,
    )
    df = df[df["entity_type"] != "unknown"]

    print("Generating Gilda candidates per document...")
    documents = []
    for doc_id, group in tqdm(df.groupby("don_article"), desc="Documents"):
        organisms = benchmarker._get_organism_priority(doc_id)
        doc = DocumentExample(doc_id=str(doc_id))
        seen_texts: dict[str, MentionExample] = {}
        for _, row in group.iterrows():
            text = row["text"]
            if text not in seen_texts:
                candidates = grounder.ground(text, organisms=organisms)
                mention = MentionExample(
                    text=text,
                    entity_type=row["entity_type"],
                    candidates=candidates,
                    gold_curies=set(row["obj"]),
                    gold_synonyms=set(row["obj_synonyms"]),
                )
                for i, cand in enumerate(candidates):
                    curie = f"{cand.term.db}:{cand.term.id}"
                    if curie in mention.gold_synonyms:
                        mention.gold_index = i
                        break
                seen_texts[text] = mention
            else:
                existing = seen_texts[text]
                existing.gold_curies.update(row["obj"])
                existing.gold_synonyms.update(row["obj_synonyms"])
                if existing.gold_index is None:
                    for i, cand in enumerate(existing.candidates):
                        curie = f"{cand.term.db}:{cand.term.id}"
                        if curie in existing.gold_synonyms:
                            existing.gold_index = i
                            break
        doc.mentions = list(seen_texts.values())
        if doc.mentions:
            documents.append(doc)

    print(f"Loaded {len(documents)} documents with "
          f"{sum(len(d.mentions) for d in documents)} unique mentions.")
    return documents


def _classify_entity_type(obj: list[str], obj_synonyms: set[str]) -> str:
    """Classify entity type (for the BioID corpus, just distinguishing Human
    vs Nonhuman Gene)
    """
    etype = _get_entity_type(obj)
    if etype == "Gene":
        if any(s.startswith("HGNC") for s in obj_synonyms):
            return "Human Gene"
        return "Nonhuman Gene"
    return etype


# For the train/val/test split

def split_by_document(
    examples: list[DocumentExample],
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[list[DocumentExample], list[DocumentExample], list[DocumentExample]]:
    """Split at document level for joint disambiguation
    """
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train)
    n_val = int(n * val)
    train_docs = [examples[i] for i in indices[:n_train]]
    val_docs = [examples[i] for i in indices[n_train : n_train + n_val]]
    test_docs = [examples[i] for i in indices[n_train + n_val :]]
    return train_docs, val_docs, test_docs



def report_statistics(examples: list[DocumentExample]) -> dict:
    """Print and return summary statistics on the corpus
    """
    n_docs = len(examples)
    n_mentions = sum(len(d.mentions) for d in examples)
    type_counts = defaultdict(int)
    n_with_candidates = 0
    n_gold_in_candidates = 0
    total_candidates = 0

    for doc in examples:
        for m in doc.mentions:
            type_counts[m.entity_type] += 1
            if m.candidates:
                n_with_candidates += 1
                total_candidates += len(m.candidates)
            if m.gold_index is not None:
                n_gold_in_candidates += 1

    stats = {
        "n_docs": n_docs,
        "n_mentions": n_mentions,
        "entity_types": dict(type_counts),
        "has_candidates_rate": n_with_candidates / n_mentions if n_mentions else 0,
        "gold_in_candidates_rate": n_gold_in_candidates / n_mentions if n_mentions else 0,
        "avg_candidates": total_candidates / n_with_candidates if n_with_candidates else 0,
    }

    print(f"Documents: {n_docs}")
    print(f"Mentions: {n_mentions}")
    print(f"Has candidates: {n_with_candidates}/{n_mentions} "
          f"({stats['has_candidates_rate']:.1%})")
    print(f"Gold in candidates: {n_gold_in_candidates}/{n_mentions} "
          f"({stats['gold_in_candidates_rate']:.1%})")
    print(f"Avg candidates per mention: {stats['avg_candidates']:.1f}")
    print("Entity type distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    return stats
