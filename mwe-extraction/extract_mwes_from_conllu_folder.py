"""
Extract MWEs from a folder of CoNLL-U (.conllu) files and write JSON outputs.

MWEs extracted (UD types):
- flat
- flat:foreign
- fixed
- compound:prt
- compound

Outputs:
- Default: one JSON per input file
- Optional: one aggregated JSON for the whole folder (--aggregate)

JSON schema (per file):
{
  "source_file": "example.conllu",
  "mwes": [
    {"mwe": "gene expression", "type": "compound", "pos_tags": "NOUN NOUN", "count": 12},
    ...
  ],
  "stats": {
    "sentences_total": 1234,
    "sentences_skipped": 3
  }
}

Examples:
  # One JSON per file
  python extract_mwes_from_conllu_folder.py --input_dir data/conllu --output_dir data/mwe_json

  # Aggregate all .conllu into a single JSON
  python extract_mwes_from_conllu_folder.py --input_dir data/conllu --output_dir data/mwe_json --aggregate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pyconll

LOGGER = logging.getLogger("mwe-extractor")


# ----------------------------
# Utilities
# ----------------------------

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def safe_int(value) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return -1


def norm_key(s: str) -> str:
    return (s or "").strip().lower()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mwe_dict_to_sorted_list(mwe_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        mwe_dict.values(),
        key=lambda d: (-int(d["count"]), str(d["mwe"]))
    )


def write_json(obj: Any, output_path: Path, indent: int = 2) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


# ----------------------------
# Sentence indexing helpers
# ----------------------------

def index_sentence(sentence) -> Tuple[Dict[int, object], List[int]]:
    """
    Build:
      - id2tok: map int token id -> token object
      - ids: sorted list of token ids
    Skips tokens with non-integer ids.
    """
    id2tok: Dict[int, object] = {}
    ids: List[int] = []
    for tok in sentence:
        tid = safe_int(tok.id)
        if tid < 1:
            continue
        id2tok[tid] = tok
        ids.append(tid)
    ids.sort()
    return id2tok, ids


def add_mwe(mwe_dict: Dict[str, Dict[str, Any]], mwe: str, mwe_type: str, pos_tags: str) -> None:
    key = norm_key(mwe)
    if not key:
        return
    if key not in mwe_dict:
        mwe_dict[key] = {"mwe": mwe, "type": mwe_type, "pos_tags": pos_tags, "count": 1}
    else:
        mwe_dict[key]["count"] = int(mwe_dict[key]["count"]) + 1


# ----------------------------
# Extractors
# ----------------------------

def extract_flat(sentence, mwe_dict: Dict[str, Dict[str, Any]]) -> None:
    id2tok, ids = index_sentence(sentence)
    groups: Dict[int, List[int]] = {}

    for tid in ids:
        tok = id2tok[tid]
        if tok.deprel == "flat":
            head_id = safe_int(tok.head)
            if head_id in id2tok:
                groups.setdefault(head_id, []).append(tid)

    for head_id, dep_ids in groups.items():
        span_ids = [head_id] + sorted(dep_ids)
        forms = [id2tok[i].form for i in span_ids]
        pos = [(id2tok[i].upos or "UNKNOWN") for i in span_ids]
        add_mwe(mwe_dict, " ".join(forms), "flat", " ".join(pos))


def extract_flat_foreign(sentence, mwe_dict: Dict[str, Dict[str, Any]]) -> None:
    id2tok, ids = index_sentence(sentence)
    groups: Dict[int, List[int]] = {}

    for tid in ids:
        tok = id2tok[tid]
        if tok.deprel == "flat:foreign":
            head_id = safe_int(tok.head)
            if head_id in id2tok:
                groups.setdefault(head_id, []).append(tid)

    for head_id, dep_ids in groups.items():
        span_ids = [head_id] + sorted(dep_ids)
        forms = [id2tok[i].form for i in span_ids]
        pos = [(id2tok[i].upos or "UNKNOWN") for i in span_ids]
        add_mwe(mwe_dict, " ".join(forms), "flat:foreign", " ".join(pos))


def extract_fixed(sentence, mwe_dict: Dict[str, Dict[str, Any]]) -> None:
    id2tok, ids = index_sentence(sentence)
    groups: Dict[int, List[int]] = {}

    for tid in ids:
        tok = id2tok[tid]
        if tok.deprel == "fixed":
            head_id = safe_int(tok.head)
            if head_id in id2tok:
                groups.setdefault(head_id, []).append(tid)

    for head_id, dep_ids in groups.items():
        span_ids = [head_id] + sorted(dep_ids)
        forms = [id2tok[i].form for i in span_ids]
        pos = [(id2tok[i].upos or "UNKNOWN") for i in span_ids]
        add_mwe(mwe_dict, " ".join(forms), "fixed", " ".join(pos))


def extract_compound_prt(sentence, mwe_dict: Dict[str, Dict[str, Any]]) -> None:
    id2tok, ids = index_sentence(sentence)
    groups: Dict[int, List[int]] = {}

    for tid in ids:
        tok = id2tok[tid]
        if tok.deprel == "compound:prt":
            head_id = safe_int(tok.head)
            if head_id in id2tok:
                groups.setdefault(head_id, []).append(tid)

    for head_id, dep_ids in groups.items():
        span_ids = sorted([head_id] + dep_ids)
        forms = [id2tok[i].form for i in span_ids]
        pos = [(id2tok[i].upos or "UNKNOWN") for i in span_ids]
        add_mwe(mwe_dict, " ".join(forms), "compound:prt", " ".join(pos))


def extract_compound_span(sentence, mwe_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    Extract compounds with a span heuristic.

    Heuristic:
    - For each token whose deprel contains 'compound' (but not compound:prt),
      take its head as the other boundary.
    - Include all tokens whose ids are between token id and head id (inclusive).
    - Keep spans of length > 1.
    """
    id2tok, ids = index_sentence(sentence)
    id_set = set(ids)

    for tid in ids:
        tok = id2tok[tid]
        if not tok.deprel:
            continue
        if "compound" in tok.deprel and tok.deprel != "compound:prt":
            head_id = safe_int(tok.head)
            if head_id not in id_set or head_id < 1:
                continue

            lo = min(tid, head_id)
            hi = max(tid, head_id)
            span_ids = [i for i in ids if lo <= i <= hi]
            if len(span_ids) <= 1:
                continue

            forms = [id2tok[i].form for i in span_ids]
            pos = [(id2tok[i].upos or "UNKNOWN") for i in span_ids]
            add_mwe(mwe_dict, " ".join(forms), "compound", " ".join(pos))


# ----------------------------
# File / folder processing
# ----------------------------

def extract_mwes_from_file(conllu_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Returns:
      mwe_dict, stats
    Where stats includes counts of total and skipped sentences.
    """
    corpus = pyconll.load_from_file(str(conllu_path))
    mwe_dict: Dict[str, Dict[str, Any]] = {}

    sentences_total = 0
    sentences_skipped = 0

    for sent_idx, sent in enumerate(corpus, start=1):
        sentences_total += 1
        try:
            extract_flat(sent, mwe_dict)
            extract_flat_foreign(sent, mwe_dict)
            extract_fixed(sent, mwe_dict)
            extract_compound_prt(sent, mwe_dict)
            extract_compound_span(sent, mwe_dict)
        except Exception as e:
            sentences_skipped += 1
            LOGGER.exception(
                "Skipping sentence %d in %s due to error: %s",
                sent_idx,
                conllu_path.name,
                e,
            )
            continue

    stats = {
        "sentences_total": sentences_total,
        "sentences_skipped": sentences_skipped,
    }
    return mwe_dict, stats


def merge_dicts(global_dict: Dict[str, Dict[str, Any]], local_dict: Dict[str, Dict[str, Any]]) -> None:
    for k, v in local_dict.items():
        if k not in global_dict:
            global_dict[k] = dict(v)
        else:
            global_dict[k]["count"] = int(global_dict[k]["count"]) + int(v["count"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract MWEs from a folder of .conllu files into JSON.")
    ap.add_argument("--input_dir", required=True, type=str, help="Folder containing .conllu files")
    ap.add_argument("--output_dir", required=True, type=str, help="Folder to write JSON outputs")
    ap.add_argument("--glob", default="*.conllu", type=str, help="Glob pattern (default: *.conllu)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files")
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Write a single aggregated JSON for all input files (also writes per-file JSON if --per_file is set).",
    )
    ap.add_argument(
        "--per_file",
        action="store_true",
        help="Write one JSON per input file (default behaviour if --aggregate is not set).",
    )
    ap.add_argument("--agg_name", default="aggregated_extracted_MWEs.json", type=str, help="Filename for aggregated JSON")
    ap.add_argument("--indent", default=2, type=int, help="JSON indent level (default: 2)")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv)")
    args = ap.parse_args()

    setup_logging(args.verbose)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        LOGGER.error("Input directory does not exist or is not a directory: %s", in_dir)
        return 2

    ensure_dir(out_dir)

    conllu_files = sorted(in_dir.glob(args.glob))
    if not conllu_files:
        LOGGER.error("No files matched %r in %s", args.glob, in_dir)
        return 2

    # Default behavior: per-file unless user asked only for aggregate
    per_file = args.per_file or not args.aggregate

    global_dict: Dict[str, Dict[str, Any]] = {}
    global_stats = {
        "files_total": len(conllu_files),
        "files_processed": 0,
        "files_failed": 0,
        "sentences_total": 0,
        "sentences_skipped": 0,
    }

    skipped_writes = 0

    for i, conllu_path in enumerate(conllu_files, start=1):
        LOGGER.info("[%d/%d] Processing %s", i, len(conllu_files), conllu_path.name)
        try:
            mwe_dict, stats = extract_mwes_from_file(conllu_path)
            global_stats["files_processed"] += 1
            global_stats["sentences_total"] += stats["sentences_total"]
            global_stats["sentences_skipped"] += stats["sentences_skipped"]
        except Exception as e:
            global_stats["files_failed"] += 1
            LOGGER.exception("Failed on file %s: %s", conllu_path.name, e)
            continue

        if per_file:
            out_path = out_dir / f"{conllu_path.stem}_extracted_MWEs.json"
            if out_path.exists() and not args.overwrite:
                skipped_writes += 1
                LOGGER.info("Skipping write (exists): %s", out_path.name)
            else:
                payload = {
                    "source_file": conllu_path.name,
                    "mwes": mwe_dict_to_sorted_list(mwe_dict),
                    "stats": stats,
                }
                write_json(payload, out_path, indent=args.indent)

        if args.aggregate:
            merge_dicts(global_dict, mwe_dict)

    if args.aggregate:
        agg_path = out_dir / args.agg_name
        if agg_path.exists() and not args.overwrite:
            LOGGER.warning("Aggregated JSON exists and --overwrite not set: %s", agg_path)
        else:
            payload = {
                "source": str(in_dir),
                "files": [p.name for p in conllu_files],
                "mwes": mwe_dict_to_sorted_list(global_dict),
                "stats": global_stats,
            }
            write_json(payload, agg_path, indent=args.indent)

    LOGGER.warning(
        "Done. files=%d processed=%d failed=%d skipped_writes=%d sentences_total=%d sentences_skipped=%d output_dir=%s",
        global_stats["files_total"],
        global_stats["files_processed"],
        global_stats["files_failed"],
        skipped_writes,
        global_stats["sentences_total"],
        global_stats["sentences_skipped"],
        out_dir,
    )
    return 0 if global_stats["files_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
