"""
Extract Bioinformatics-only articles from allofplos, then
write the main IMRaD(+Abstract+Conclusion) sections to per-section folders as .txt files.

- Filters input XMLs by <subject> containing (case-insensitive) "bioinformatics"
- Extracts these sections (when present):
  Abstract, Introduction, Materials and methods, Results, Discussion, Conclusions
- Outputs: one .txt per (article, section), placed in output_root/<SectionName>/

Usage:
  python corpus_build.py /path/to/xml_folder /path/to/output_folder
  python corpus_build.py /path/to/xml_folder /path/to/output_folder --subject bioinformatics  #change subject if looking to compile a corpus of different domain
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Prefer lxml for robustness; fall back to stdlib ElementTree.
try:
    from lxml import etree as ET  # type: ignore
    LXML = True
except Exception:
    import xml.etree.ElementTree as ET  # type: ignore
    LXML = False


# ----------------------------
# Text cleaning / normalization
# ----------------------------

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")

def normalize_text(s: str) -> str:
    """Collapse excessive whitespace while keeping paragraph breaks."""
    s = s.replace("\u00a0", " ")  # non-breaking space
    # Normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Trim trailing spaces on lines
    s = "\n".join(line.strip() for line in s.split("\n"))
    # Collapse runs of spaces/tabs
    s = _WS_RE.sub(" ", s)
    # Collapse excessive blank lines
    s = _NL_RE.sub("\n\n", s)
    return s.strip()


def safe_filename(s: str, max_len: int = 200) -> str:
    """Make a filesystem-safe filename."""
    s = s.strip()
    s = s.replace("/", "_")
    s = re.sub(r"[^\w.\-+]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


# ----------------------------
# XML helpers
# ----------------------------

def parse_xml(path: Path):
    """Parse an XML file and return its root element."""
    if LXML:
        parser = ET.XMLParser(recover=True, huge_tree=True, remove_comments=False)
        tree = ET.parse(str(path), parser)
        return tree.getroot()
    else:
        tree = ET.parse(str(path))
        return tree.getroot()


def elem_text(e) -> str:
    """All descendant text, concatenated."""
    if e is None:
        return ""
    return "".join(e.itertext())


def get_article_id(root, fallback_stem: str) -> str:
    """
    Prefer DOI; else PMID; else publisher-id; else fallback to filename stem.
    """
    # JATS typically: front/article-meta/article-id[@pub-id-type="doi"]
    for pub_id_type in ("doi", "pmid", "publisher-id", "pmc", "pii"):
        xp = f'.//article-meta/article-id[@pub-id-type="{pub_id_type}"]'
        try:
            node = root.find(xp)
        except Exception:
            node = None
        if node is not None:
            val = elem_text(node).strip()
            if val:
                return val
    return fallback_stem


def has_subject(root, subject_kw: str) -> bool:
    """
    Return True if any <subject> element contains subject_kw (case-insensitive).
    """
    kw = subject_kw.strip().lower()
    if not kw:
        return True

    try:
        subjects = root.findall(".//subject")
    except Exception:
        subjects = []

    for s in subjects:
        txt = elem_text(s).strip().lower()
        if kw in txt:
            return True
    return False


# ----------------------------
# Section extraction
# ----------------------------

SECTION_DIRS = {
    "Abstract": "Abstract",
    "Introduction": "Introduction",
    "Materials and methods": "Materials_and_methods",
    "Results": "Results",
    "Discussion": "Discussion",
    "Conclusions": "Conclusions",
}

def extract_abstract(root) -> Optional[str]:
    """
    Extract the *main* abstract (exclude author/editor summaries when possible).
    For PLOS/JATS, main abstract is usually <article-meta><abstract> with no abstract-type.
    """
    # Prefer abstract with no abstract-type
    abs_nodes = root.findall(".//article-meta/abstract")
    main_abs = None
    for a in abs_nodes:
        atype = a.get("abstract-type") if hasattr(a, "get") else None
        if atype is None:
            main_abs = a
            break

    if main_abs is None and abs_nodes:
        # fallback to the first abstract
        main_abs = abs_nodes[0]

    if main_abs is None:
        return None

    # Extract by paragraphs to keep some structure
    ps = main_abs.findall(".//p")
    if ps:
        text = "\n\n".join(normalize_text(elem_text(p)) for p in ps if normalize_text(elem_text(p)))
    else:
        text = normalize_text(elem_text(main_abs))

    return text or None


def sec_label(sec) -> Tuple[str, str]:
    """
    Return (sec_type, title_text) lowercased and stripped.
    """
    sec_type = ""
    title = ""
    try:
        sec_type = (sec.get("sec-type") or "").strip().lower()
    except Exception:
        sec_type = ""
    try:
        t = sec.find("./title")
        title = elem_text(t).strip().lower() if t is not None else ""
    except Exception:
        title = ""
    return sec_type, title


def classify_section(sec) -> Optional[str]:
    """
    Map a <sec> to one of the target sections, using sec-type first then title heuristics.
    """
    sec_type, title = sec_label(sec)

    # Normalize common JATS/PLOS patterns
    st = sec_type.replace("_", " ").strip()

    # Strong matches by sec-type
    if st in {"intro", "introduction"}:
        return "Introduction"
    if "materials|methods" in st or st in {"methods", "materials and methods", "materials", "method"}:
        return "Materials and methods"
    if st == "results":
        return "Results"
    if st in {"discussion", "disc"}:
        return "Discussion"
    if st.startswith("conclu"):
        # Some corpora misuse sec-type="conclusions" for a Discussion section; use title to disambiguate.
        if "discussion" in title and "conclusion" not in title:
            return "Discussion"
        return "Conclusions"

    # Heuristics from the title when sec-type is absent/odd
    if title:
        if "introduction" in title:
            return "Introduction"
        if "materials and methods" in title or "methods" in title or "methodology" in title:
            return "Materials and methods"
        if title == "results" or title.startswith("results"):
            return "Results"
        if "discussion" in title:
            return "Discussion"
        if "conclusion" in title or title.startswith("concluding"):
            return "Conclusions"

    return None


def extract_sec_text(sec) -> Optional[str]:
    """
    Extract readable text from a section, excluding the section title node itself.
    Keeps paragraph boundaries.
    """
    # Prefer paragraph-based extraction for clean output
    ps = sec.findall(".//p")
    parts: List[str] = []

    for p in ps:
        t = normalize_text(elem_text(p))
        if t:
            parts.append(t)

    # Also capture list items if paragraphs are missing or lists are used heavily
    if not parts:
        lis = sec.findall(".//list-item")
        for li in lis:
            t = normalize_text(elem_text(li))
            if t:
                parts.append(t)

    if parts:
        return "\n\n".join(parts).strip() or None

    # Fallback: all text, but try to avoid duplicating the title if present
    # (remove the first occurrence of the title text if it starts the section text)
    raw = normalize_text(elem_text(sec))
    tnode = sec.find("./title")
    title = normalize_text(elem_text(tnode)) if tnode is not None else ""
    if title and raw.lower().startswith(title.lower()):
        raw = raw[len(title):].lstrip()
    return raw or None


def extract_imrad_sections(root) -> Dict[str, str]:
    """
    Return dict: {section_name: section_text} for the target sections.
    """
    out: Dict[str, str] = {}

    # Abstract
    abs_txt = extract_abstract(root)
    if abs_txt:
        out["Abstract"] = abs_txt

    # Body sections (focus on top-level secs; still harvest text from nested subsecs)
    body = root.find(".//body")
    if body is None:
        return out

    for sec in body.findall("./sec"):
        label = classify_section(sec)
        if not label:
            continue

        txt = extract_sec_text(sec)
        if not txt:
            continue

        # If multiple top-level secs map to the same label, concatenate with a blank line.
        if label in out:
            out[label] = normalize_text(out[label] + "\n\n" + txt)
        else:
            out[label] = txt

    return out


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Folder containing XML files")
    ap.add_argument("output_dir", type=str, help="Output folder (will be created if missing)")
    ap.add_argument("--subject", type=str, default="bioinformatics",
                    help='Subject keyword to filter on (default: "bioinformatics")')
    ap.add_argument("--glob", type=str, default="*.xml", help="Glob pattern for XML files (default: *.xml)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create section directories
    for d in SECTION_DIRS.values():
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    xml_paths = sorted(in_dir.glob(args.glob))
    if not xml_paths:
        raise SystemExit(f"No files matched {args.glob} in {in_dir}")

    kept = 0
    written = 0
    parse_errors = 0

    for xml_path in xml_paths:
        try:
            root = parse_xml(xml_path)
        except Exception as e:
            parse_errors += 1
            print(f"[WARN] Failed to parse {xml_path.name}: {e}")
            continue

        if not has_subject(root, args.subject):
            continue

        kept += 1

        art_id = get_article_id(root, xml_path.stem)
        art_id_safe = safe_filename(art_id) or safe_filename(xml_path.stem)

        sections = extract_imrad_sections(root)
        if not sections:
            print(f"[INFO] No target sections found in {xml_path.name}")
            continue

        for sec_name, sec_text in sections.items():
            sec_dir = SECTION_DIRS.get(sec_name, safe_filename(sec_name))
            out_path = out_dir / sec_dir / f"{art_id_safe}__{safe_filename(sec_name)}.txt"

            if out_path.exists() and not args.overwrite:
                continue

            out_path.write_text(sec_text + "\n", encoding="utf-8")
            written += 1

    print("Done.")
    print(f"  XML files scanned: {len(xml_paths)}")
    print(f"  Bioinformatics-matching files kept: {kept}")
    print(f"  .txt files written: {written}")
    if parse_errors:
        print(f"  Parse errors: {parse_errors}")


if __name__ == "__main__":
    main()
