"""
PyMUSAS MWE extraction over a folder of .txt files -> JSON outputs (with optional chunking).

ALL OPTIONS (CLI flags)
-----------------------
Required:
  --input_dir PATH           Folder containing .txt files
  --output_dir PATH          Folder to write JSON outputs

Input selection:
  --glob PATTERN             Glob pattern for input files (default: *.txt)

Models / hardware:
  --spacy_model NAME         Base spaCy model (default: en_core_web_sm)
  --pymusas_model NAME       PyMUSAS pipeline model (default: en_dual_none_contextual)
  --use_gpu                  Prefer GPU (if available)

Chunking:
  --chunk_size INT           Chunk size in characters (default: 100000)
  --overlap INT              Overlap (left context) in characters (default: 200)
  --no_chunking              Disable chunking entirely (process full file in one pass)

Output:
  --aggregate                If set: write ONE aggregated JSON for the whole folder
                             Else (default): write ONE JSON PER INPUT FILE
  --agg_name FILENAME        Name for aggregated JSON (default: aggregated_usas_mwes.json)
  --indent INT               JSON indent level (default: 2)
  --overwrite                Overwrite existing JSON outputs

Logging:
  -v, --verbose              Increase verbosity (-v, -vv)

What it exports
---------------
ONLY MWEs (no token-level rows, no frequency printing).

Per-file JSON schema
{
  "source_file": "myfile.txt",
  "chunking": {"enabled": true, "chunk_size_chars": 100000, "overlap_chars": 200},
  "stats": {"chunks_total": 3, "chunks_skipped": 0, "mwes_total": 123},
  "mwes": [
    {"text": "in vitro", "pos": "ADP", "start": 10, "end": 12, "usas_tags": [...],
     "chunk_index": 0, "text_char_start": 0, "text_char_end": 99512},
    ...
  ]
}

Aggregate JSON schema
{
  "source": "/path/to/input_dir",
  "files": ["a.txt", "b.txt", ...],
  "chunking": {"enabled": true, "chunk_size_chars": 100000, "overlap_chars": 200},
  "stats": {
    "files_total": 10,
    "files_processed": 10,
    "files_failed": 0,
    "chunks_total": 120,
    "chunks_skipped": 2,
    "mwes_total": 4567
  },
  "mwes": [
    {"source_file": "a.txt", "text": "...", ...},
    {"source_file": "b.txt", "text": "...", ...}
  ]
}

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import spacy

LOGGER = logging.getLogger("pymusas-mwe-folder")


# ----------------------------
# Logging / JSON
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


def write_json(obj: Any, out_path: Path, indent: int = 2) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


# ----------------------------
# Chunking
# ----------------------------

def iter_text_chunks(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Return list of (start_char, end_char, chunk_text).
    Tries to end chunks on whitespace near boundaries; overlap adds some left context.
    """
    n = len(text)
    chunks: List[Tuple[int, int, str]] = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)

        # Prefer splitting on whitespace shortly after the boundary (up to +500 chars)
        if end < n:
            window_end = min(end + 500, n)
            ws_pos = text.rfind(" ", end, window_end)
            if ws_pos != -1 and ws_pos > start:
                end = ws_pos + 1

        chunk_start = max(0, start - overlap) if chunks else start
        chunk_text = text[chunk_start:end]
        chunks.append((chunk_start, end, chunk_text))
        start = end

    return chunks


# ----------------------------
# spaCy + PyMUSAS pipeline
# ----------------------------

def build_pipeline(spacy_model: str, pymusas_model: str, use_gpu: bool) -> "spacy.Language":
    if use_gpu:
        activated = spacy.prefer_gpu()
        LOGGER.info("spaCy GPU preference: %s", activated)

    nlp = spacy.load(spacy_model, exclude=["parser", "ner"])
    english_tagger_pipeline = spacy.load(pymusas_model)
    nlp.add_pipe("pymusas_rule_based_tagger", source=english_tagger_pipeline)
    return nlp


def tags_to_list(tags) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, (list, tuple)):
        return [str(t) for t in tags]
    return [str(tags)]


def extract_mwes_from_chunk(nlp: "spacy.Language", chunk_text: str, chunk_index: int) -> List[Dict[str, Any]]:
    doc = nlp(chunk_text)
    mwes_out: List[Dict[str, Any]] = []

    for tok in doc:
        mwe_idxs = getattr(tok._, "pymusas_mwe_indexes", None)
        if not mwe_idxs:
            continue

        # Use the first span (matches your original code)
        try:
            start, end = mwe_idxs[0]
        except Exception:
            continue

        if (end - start) > 1:
            mwes_out.append(
                {
                    "text": tok.text,
                    "pos": tok.pos_,
                    "start": int(start),
                    "end": int(end),
                    "usas_tags": tags_to_list(getattr(tok._, "pymusas_tags", None)),
                    "chunk_index": chunk_index,
                }
            )

    return mwes_out


# ----------------------------
# File processing (MWE-only)
# ----------------------------

def process_text_file_mwes(
    nlp: "spacy.Language",
    txt_path: Path,
    chunking_enabled: bool,
    chunk_size: int,
    overlap: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (mwes, stats) for a single file.
    Skips problematic chunks and continues.
    """
    text = txt_path.read_text(encoding="utf-8", errors="replace")

    if not chunking_enabled:
        chunks = [(0, len(text), text)]
    else:
        if len(text) <= chunk_size:
            chunks = [(0, len(text), text)]
        else:
            chunks = iter_text_chunks(text, chunk_size, overlap)

    chunks_total = len(chunks)
    chunks_skipped = 0
    all_mwes: List[Dict[str, Any]] = []

    for chunk_index, (c_start, c_end, c_text) in enumerate(chunks):
        try:
            mwes = extract_mwes_from_chunk(nlp, c_text, chunk_index)
            for m in mwes:
                # Minimal provenance for boundaries/debugging
                m["text_char_start"] = c_start
                m["text_char_end"] = c_end
            all_mwes.extend(mwes)
        except Exception as e:
            chunks_skipped += 1
            LOGGER.exception("Skipping chunk %d in %s due to error: %s", chunk_index, txt_path.name, e)
            continue

    stats = {
        "chunks_total": chunks_total,
        "chunks_skipped": chunks_skipped,
        "mwes_total": len(all_mwes),
    }
    return all_mwes, stats


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract PyMUSAS MWEs from a folder of .txt files into JSON (per-file or aggregate)."
    )

    # Required
    ap.add_argument("--input_dir", required=True, type=str, help="Folder containing .txt files")
    ap.add_argument("--output_dir", required=True, type=str, help="Folder to write JSON outputs")

    # Input selection
    ap.add_argument("--glob", default="*.txt", type=str, help="Glob pattern (default: *.txt)")

    # Models / hardware
    ap.add_argument("--spacy_model", default="en_core_web_sm", type=str, help="Base spaCy model (default: en_core_web_sm)")
    ap.add_argument("--pymusas_model", default="en_dual_none_contextual", type=str, help="PyMUSAS model (default: en_dual_none_contextual)")
    ap.add_argument("--use_gpu", action="store_true", help="Prefer GPU (if available)")

    # Chunking
    ap.add_argument("--chunk_size", default=100000, type=int, help="Chunk size in characters (default: 100000)")
    ap.add_argument("--overlap", default=200, type=int, help="Overlap in characters (default: 200)")
    ap.add_argument("--no_chunking", action="store_true", help="Disable chunking (process each file in one pass)")

    # Output (exactly two options)
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="If set: write ONE aggregated JSON for the whole folder. Otherwise: write ONE JSON PER INPUT FILE.",
    )
    ap.add_argument("--agg_name", default="aggregated_usas_mwes.json", type=str, help="Name for aggregated JSON (default: aggregated_usas_mwes.json)")
    ap.add_argument("--indent", default=2, type=int, help="JSON indent level (default: 2)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON outputs")

    # Logging
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    args = ap.parse_args()

    setup_logging(args.verbose)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        LOGGER.error("Input directory does not exist or is not a directory: %s", in_dir)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted([p for p in in_dir.glob(args.glob) if p.is_file()])
    if not txt_files:
        LOGGER.error("No files matched %r in %s", args.glob, in_dir)
        return 2

    chunking_enabled = not args.no_chunking

    nlp = build_pipeline(args.spacy_model, args.pymusas_model, args.use_gpu)

    stats = {
        "files_total": len(txt_files),
        "files_processed": 0,
        "files_failed": 0,
        "chunks_total": 0,
        "chunks_skipped": 0,
        "mwes_total": 0,
    }

    if args.aggregate:
        all_mwes: List[Dict[str, Any]] = []

        for i, txt_path in enumerate(txt_files, start=1):
            LOGGER.info("[%d/%d] Processing: %s", i, len(txt_files), txt_path.name)
            try:
                mwes, file_stats = process_text_file_mwes(
                    nlp=nlp,
                    txt_path=txt_path,
                    chunking_enabled=chunking_enabled,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                )
            except Exception as e:
                stats["files_failed"] += 1
                LOGGER.exception("Failed processing %s: %s", txt_path.name, e)
                continue

            stats["files_processed"] += 1
            stats["chunks_total"] += int(file_stats["chunks_total"])
            stats["chunks_skipped"] += int(file_stats["chunks_skipped"])
            stats["mwes_total"] += int(file_stats["mwes_total"])

            for m in mwes:
                m["source_file"] = txt_path.name
            all_mwes.extend(mwes)

        agg_path = out_dir / args.agg_name
        if agg_path.exists() and not args.overwrite:
            LOGGER.error("Aggregated JSON exists and --overwrite not set: %s", agg_path)
            return 2

        payload = {
            "source": str(in_dir),
            "files": [p.name for p in txt_files],
            "chunking": {
                "enabled": chunking_enabled,
                "chunk_size_chars": args.chunk_size,
                "overlap_chars": args.overlap,
            },
            "stats": stats,
            "mwes": all_mwes,
        }
        write_json(payload, agg_path, indent=args.indent)

    else:
        for i, txt_path in enumerate(txt_files, start=1):
            LOGGER.info("[%d/%d] Processing: %s", i, len(txt_files), txt_path.name)
            out_path = out_dir / f"{txt_path.stem}_usas_mwes.json"

            if out_path.exists() and not args.overwrite:
                LOGGER.info("[%d/%d] Skipping (exists): %s", i, len(txt_files), out_path.name)
                continue

            try:
                mwes, file_stats = process_text_file_mwes(
                    nlp=nlp,
                    txt_path=txt_path,
                    chunking_enabled=chunking_enabled,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                )
            except Exception as e:
                stats["files_failed"] += 1
                LOGGER.exception("Failed processing %s: %s", txt_path.name, e)
                continue

            stats["files_processed"] += 1
            stats["chunks_total"] += int(file_stats["chunks_total"])
            stats["chunks_skipped"] += int(file_stats["chunks_skipped"])
            stats["mwes_total"] += int(file_stats["mwes_total"])

            payload = {
                "source_file": txt_path.name,
                "chunking": {
                    "enabled": chunking_enabled,
                    "chunk_size_chars": args.chunk_size,
                    "overlap_chars": args.overlap,
                },
                "stats": file_stats,
                "mwes": mwes,
            }
            write_json(payload, out_path, indent=args.indent)

    LOGGER.warning(
        "Done. files=%d processed=%d failed=%d chunks_total=%d chunks_skipped=%d mwes_total=%d output_dir=%s",
        stats["files_total"],
        stats["files_processed"],
        stats["files_failed"],
        stats["chunks_total"],
        stats["chunks_skipped"],
        stats["mwes_total"],
        out_dir,
    )
    return 0 if stats["files_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
