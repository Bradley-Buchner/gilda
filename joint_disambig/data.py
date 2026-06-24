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


@dataclass
class MentionExample:
    """For a single mention's candidates and ground truth label.
    """
    text: str
    entity_type: str
    candidates: list
    gold_curies: set = field(default_factory=set)
    gold_synonyms: set = field(default_factory=set)
    gold_index: Optional[int] = None
    offsets: Optional[list] = None
    source_datasets: set = field(default_factory=set)


@dataclass
class DocumentExample:
    """For all mentions from one document, forming one training example
    """
    doc_id: str
    mentions: list = field(default_factory=list)
    source: Optional[str] = None
    split: Optional[str] = None


_normalize_id = BioIDBenchmarker._normalize_id
_normalize_ids = BioIDBenchmarker._normalize_ids
_get_entity_type = BioIDBenchmarker._get_entity_type

_mesh_chebi_crosswalk: Optional[dict] = None

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


def assign_gold_index(mention: MentionExample) -> None:
    """Set mention.gold_index to the position of the first candidate
    whose curie is in gold_synonyms, or None if no candidate matches.
    """
    if mention.gold_index is not None:
        return
    for i, cand in enumerate(mention.candidates):
        if f"{cand.term.db}:{cand.term.id}" in mention.gold_synonyms:
            mention.gold_index = i
            return


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
                assign_gold_index(mention)
                seen_texts[text] = mention
            else:
                existing = seen_texts[text]
                existing.gold_curies.update(row["obj"])
                existing.gold_synonyms.update(row["obj_synonyms"])
                if existing.gold_index is None:
                    assign_gold_index(existing)
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


def make_splits(docs):
    """Partition a corpus into train, validation, and test sets by each document's
    assigned split.
    """
    train = [d for d in docs if d.split == "train"]
    val = [d for d in docs if d.split == "validation"]
    test = [d for d in docs if d.split == "test"]
    unsplit = [d for d in docs if d.split not in ("train", "validation", "test")]
    if unsplit:
        train_extra, val_extra, test_extra = split_by_document(unsplit)
        train.extend(train_extra)
        val.extend(val_extra)
        test.extend(test_extra)
    return train, val, test



def _normalize_bigbio_curie(curie: str) -> list[str]:
    """Convert a BigBio gold curie to Gilda convention. Simple string
    formatting.
    """
    prefix, _, identifier = curie.partition(":")
    identifier = identifier.strip()
    if not identifier:
        return []
    if prefix == "MESH":
        return [f"MESH:{identifier}"]  # MESH already native to Gilda
    if prefix == "OMIM":
        return [f"OMIM:{identifier}"]
    if prefix == "NCBIGene":
        return [f"NCBI gene:{identifier}"]
    if prefix in ("CHEBI", "GO"):
        return [_normalize_id(f"{prefix}:{identifier}")]
    if prefix == "UMLS":
        return []  # UMLS prefixes are handled in expand_gold_curies
    return [curie]

def build_mesh_chebi_crosswalk(cache_path=None) -> dict:
    """Build a mapping from MeSH id to CHEBI curies by inverting
    bio_ontology's CHEBI to MeSH xrefs. Build once and cache.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return {k: set(v) for k, v in json.load(f).items()}
    from indra.ontology.bio import bio_ontology
    bio_ontology.initialize()
    inverse = defaultdict(set)
    for node in bio_ontology.nodes:
        if not node.startswith("CHEBI:"):
            continue
        ns, _, rid = node.partition(":")
        for prefix, ident in bio_ontology.get_mappings(ns, rid):
            if prefix == "MESH":
                inverse[ident].add(f"CHEBI:{rid}")
    out = {k: sorted(v) for k, v in inverse.items()}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f)

    return {k: set(v) for k, v in out.items()}

def _get_mesh_chebi_crosswalk() -> dict:
    global _mesh_chebi_crosswalk
    if _mesh_chebi_crosswalk is None:
        _mesh_chebi_crosswalk = build_mesh_chebi_crosswalk(
            cache_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "mesh_chebi_crosswalk.json"))
    return _mesh_chebi_crosswalk


# Dict to specify allowed namespaces and handle the MSH->MESH translation
_UMLS_SAB_MAP = {"MSH": "MESH", "HGNC": "HGNC", "OMIM": "OMIM", "GO": "GO"}

def build_umls_crosswalk(mrconso_path, cache_path=None, langs=("ENG",)):
    """Build a mapping from CUI ids to Gilda-formatted xref curies from the UMLS Metathesaurus.
    MRCONSO.RRF is read to build the mapping once, then it's cached.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return {k: set(v) for k, v in json.load(f).items()}
    crosswalk = defaultdict(set)
    with open(mrconso_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            cui, lat, sab, code = p[0], p[1], p[11], p[13]
            if langs and lat not in langs:
                continue
            ns = _UMLS_SAB_MAP.get(sab)
            if not ns:
                continue
            bare = code.split(":")[-1]
            seeds = {"MESH": f"MESH:{bare}", "HGNC": f"HGNC:{bare}",
                    "GO": f"GO:GO:{bare}", "OMIM": f"OMIM:{bare}"}
            seed = seeds[ns]
            crosswalk[cui].add(seed)
    crosswalk = {k: sorted(v) for k, v in crosswalk.items()}
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(crosswalk, f)
        print(f"UMLS crosswalk: {len(crosswalk)} CUIs -> {cache_path}")
    return {k: set(v) for k, v in crosswalk.items()}


def expand_gold_curies(db_ids, benchmarker=None, umls_crosswalk=None) -> set[str]:
    """Expand a BigBio mention's gold db ids into synonyms using the
    BioIDBenchmarker's get_synonym_set method.

    Params:
    -------
    db_ids: list[str]
        list of string db ids
    benchmarker:
        BioIDBenchmarker (created lazily)
    umls_crosswalk: dict[str, set[str]]
        Dict with CUI keys and sets of Gilda-formatted curies as values

    Returns:
    --------
    Set of curies formatted in the Gilda convention
    """
    if benchmarker is None:
        benchmarker = _get_benchmarker()
    mesh_chebi = _get_mesh_chebi_crosswalk()
    seeds = []
    for curie in db_ids:
        if curie.startswith("UMLS:") and umls_crosswalk is not None:
            seeds.extend(umls_crosswalk.get(curie.split(":", 1)[1], ()))
        else:
            seeds.extend(_normalize_bigbio_curie(curie))
    # Bridge any MeSH seed to CHEBI
    for s in list(seeds):
        if s.startswith("MESH:"):
            seeds.extend(mesh_chebi.get(s.split(":", 1)[1], ()))
    if not seeds:
        return set()
    return benchmarker.get_synonym_set(seeds)


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
