"""
Parse a folder of .txt files with Stanza and export each document to CoNLL-U (.conllu).

Key features
- Processes one file at a time from an input folder
- Writes one .conllu per input file into an output folder
- Lets users choose a Stanza model package:
    * default (generic)
    * biomed presets via --biomed (maps to genia/craft)
    * or an explicit --package value

Examples
1) Default English pipeline:
   python parse_txt_folder_to_conllu.py --input_dir data/txt --output_dir data/conllu --download_if_missing

2) Biomed preset (GENIA):
   python parse_txt_folder_to_conllu.py --input_dir data/txt --output_dir data/conllu --biomed genia --download_if_missing

3) Explicit package:
   python parse_txt_folder_to_conllu.py --input_dir data/txt --output_dir data/conllu --package craft --download_if_missing

Notes
- --biomed is just a convenience wrapper around known biomed-friendly packages.
- Not all Stanza versions ship every package/model; if your chosen package isn't available,
  Stanza will raise an error during pipeline initialization.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import stanza
from stanza.utils.conll import CoNLL

LOGGER = logging.getLogger("stanza-folder-parser")


# ----------------------------
# Logging
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


# ----------------------------
# IO helpers
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_txt_files(input_dir: Path, glob_pattern: str) -> List[Path]:
    files = sorted(input_dir.glob(glob_pattern))
    return [p for p in files if p.is_file()]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ----------------------------
# Stanza pipeline helpers
# ----------------------------

BIOMED_PRESET_TO_PACKAGE = {
    # Commonly used Stanza biomedical packages (availability depends on Stanza version)
    "genia": "genia",
    "craft": "craft",
    # You can add more presets here if you use them internally.
}


def resolve_package(explicit_package: Optional[str], biomed: Optional[str]) -> Optional[str]:
    """
    Decide which Stanza package to use.
    Priority:
      1) --package (explicit)
      2) --biomed preset (mapped)
      3) None (default Stanza package)
    """
    if explicit_package:
        return explicit_package
    if biomed:
        key = biomed.strip().lower()
        if key not in BIOMED_PRESET_TO_PACKAGE:
            raise ValueError(
                f"Unknown biomed preset: {biomed}. "
                f"Valid options: {', '.join(sorted(BIOMED_PRESET_TO_PACKAGE))}"
            )
        return BIOMED_PRESET_TO_PACKAGE[key]
    return None


def build_pipeline(
    lang: str,
    processors: str,
    package: Optional[str],
    use_gpu: bool,
    download_if_missing: bool,
) -> "stanza.Pipeline":
    """
    Build and return a Stanza pipeline. Optionally downloads models first.
    """
    if download_if_missing:
        LOGGER.info("Downloading stanza models (lang=%s)...", lang)
        stanza.download(lang)

    pipeline_kwargs = {
        "lang": lang,
        "processors": processors,
        "use_gpu": use_gpu,
    }
    if package:
        pipeline_kwargs["package"] = package

    LOGGER.info(
        "Initializing Stanza pipeline: lang=%s processors=%s package=%s use_gpu=%s",
        lang,
        processors,
        package or "default",
        use_gpu,
    )
    return stanza.Pipeline(**pipeline_kwargs)


# ----------------------------
# Processing
# ----------------------------

def process_file(nlp: "stanza.Pipeline", input_path: Path, output_path: Path) -> None:
    text = read_text(input_path)
    if not text.strip():
        LOGGER.warning("Empty/whitespace-only file: %s (skipping)", input_path.name)
        return

    doc = nlp(text)
    CoNLL.write_doc2conll(doc, str(output_path))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse .txt files in a folder with Stanza and write .conllu outputs."
    )
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing .txt files")
    ap.add_argument("--output_dir", type=str, required=True, help="Folder to write .conllu files")

    ap.add_argument("--glob", type=str, default="*.txt", help="Input glob pattern (default: *.txt)")
    ap.add_argument("--lang", type=str, default="en", help="Stanza language code (default: en)")
    ap.add_argument(
        "--processors",
        type=str,
        default="tokenize,mwt,pos,lemma,depparse",
        help="Stanza processors (default: tokenize,mwt,pos,lemma,depparse)",
    )

    # Package choice
    ap.add_argument(
        "--biomed",
        type=str,
        default=None,
        choices=sorted(BIOMED_PRESET_TO_PACKAGE.keys()),
        help="Convenience biomed preset (e.g., genia, craft). Ignored if --package is set.",
    )
    ap.add_argument(
        "--package",
        type=str,
        default=None,
        help="Explicit stanza package name (overrides --biomed).",
    )

    ap.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU if available (requires appropriate PyTorch/CUDA setup).",
    )
    ap.add_argument(
        "--download_if_missing",
        action="store_true",
        help="Run stanza.download(lang) before processing.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .conllu files (default: skip if exists).",
    )
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv)")

    args = ap.parse_args()
    setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        LOGGER.error("Input dir does not exist or is not a directory: %s", input_dir)
        return 2

    ensure_dir(output_dir)

    txt_files = list_txt_files(input_dir, args.glob)
    if not txt_files:
        LOGGER.error("No input files matched %r in %s", args.glob, input_dir)
        return 2

    try:
        resolved_package = resolve_package(args.package, args.biomed)
    except ValueError as e:
        LOGGER.error(str(e))
        return 2

    nlp = build_pipeline(
        lang=args.lang,
        processors=args.processors,
        package=resolved_package,
        use_gpu=args.use_gpu,
        download_if_missing=args.download_if_missing,
    )

    total = len(txt_files)
    processed = 0
    skipped = 0
    failed = 0

    for i, in_path in enumerate(txt_files, start=1):
        out_name = in_path.with_suffix(".conllu").name
        out_path = output_dir / out_name

        if out_path.exists() and not args.overwrite:
            skipped += 1
            LOGGER.info("[%d/%d] Skipping (exists): %s", i, total, out_name)
            continue

        LOGGER.info("[%d/%d] Processing: %s", i, total, in_path.name)
        try:
            process_file(nlp, in_path, out_path)
            processed += 1
        except Exception as e:
            failed += 1
            LOGGER.exception("Failed on %s: %s", in_path.name, e)

    LOGGER.warning(
        "Done. total=%d processed=%d skipped=%d failed=%d output_dir=%s",
        total,
        processed,
        skipped,
        failed,
        output_dir,
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
