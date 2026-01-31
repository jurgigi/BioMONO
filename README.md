# MWE Identification in Bioinformatics Research Articles and Dispersion Profiling Across IMRaD

This repository contains the code used in the paper:

**“Beyond Single Words: MWE Identification in Bioinformatics Research Articles and Dispersion Profiling Across IMRaD”**

It supports two main workflows:

1. **Corpus building**: filter bioinformatics articles from a JATS XML collection and segment each article into main IMRaD sections (Abstract, Introduction, Materials and Methods, Results, Discussion, Conclusions when available).
2. **MWE extraction**: extract Multiword Expressions (MWEs) from the resulting corpus using:
   - **UD-based extraction** (via dependency parsing)
   - **USAS-based extraction** (via PyMUSAS / UCREL semantic tagging)
   - (Optionally) additional list-based / terminology resources (MeSH, AFL/ARTES) stored in `mwes-lists/`

The segmented corpus itself is distributed separately (see **Corpus access** below).

---

## Corpus access

The corpus (already segmented into IMRaD sections) is available for download here: 
https://doi.org/10.6084/m9.figshare.31215955

---

## Repository structure

```text
.
├── corpus-building/     # Python scripts to collect, filter, and structure the corpus
├── mwe-extraction/      # Python scripts to extract MWEs (UD and USAS)
├── mwes-lists/          # lists of MWEs from AFL, ARTES and the MeSH controlled vocabulary thesaurus
├── LICENSE              # Code license
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone git@github.com:jurgigi/BioMONO.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download model

**spaCy model:**

```bash
python -m spacy download en_core_web_sm
```

---

## Methods overview

### Corpus presentation

`BioMONO_en` is derived from the PLOS **allofplos** collection (JATS XML). Articles belonging to the **bioinformatics** subject are filtered and then segmented into IMRaD sections using JATS section titles / tags, producing one plain-text file per section per article.

PLOS allofplos:
https://github.com/PLOS/allofplos

---

## MWE extraction approaches

MWEs are extracted using complementary automated methods:

- **UD-based MWEs**: extracted from dependency parses using relations commonly associated with multiword constructions:
  - `compound` (incl. nominal compounds)
  - `compound:prt` (phrasal verbs)
  - `fixed` (grammaticalized fixed expressions)
  - `flat` (headless flat constructions)
  - `flat:foreign` (foreign sequences)

Parsing is performed with **Stanza**:
https://github.com/stanfordnlp/stanza

- **USAS-based MWEs**: extracted via **PyMUSAS**, which exposes UCREL’s USAS semantic resources and includes MWE tagging support:
https://github.com/UCREL/pymusas

- **MeSH / AFL / ARTES lists**: stored in `mwes-lists/` for optional list-based matching in downstream analyses.

MeSH:
https://www.nlm.nih.gov/mesh/meshhome.html

AFL:
https://www.eapfoundation.com/vocab/academic/afl/

ARTES:
https://artes.app.univ-paris-diderot.fr/

---

## How to use

### A) Corpus building (XML → IMRaD TXT)

**Goal**: from a folder of JATS XML files, keep only those whose subject contains “bioinformatics”, then extract IMRaD sections into section-specific output folders.

```bash
python corpus-building/corpus_build.py \
  /path/to/xml_folder \
  /path/to/output_imrad_txt \
  --subject bioinformatics
```

### UD-based MWEs (TXT → CoNLL-U → JSON)

#### 1) Parse section texts with Stanza (TXT → CoNLL-U)

```bash
python mwe-extraction/parse_txt_folder_to_conllu.py \
  --input_dir /path/to/output_imrad_txt/Introduction \
  --output_dir /path/to/conllu/Introduction \
  --download_if_missing \
  --use_gpu
```

Optional: use a domain package (if available in your Stanza setup):

```bash
python mwe-extraction/parse_txt_folder_to_conllu.py \
  --input_dir /path/to/output_imrad_txt/Introduction \
  --output_dir /path/to/conllu/Introduction \
  --biomed genia \
  --download_if_missing \
  --use_gpu
```

#### 2) Extract UD MWEs (CoNLL-U → JSON)

Per-file JSON outputs (default):

```bash
python mwe-extraction/extract_mwes_from_conllu_folder.py \
  --input_dir /path/to/conllu/Introduction \
  --output_dir /path/to/ud_mwes_json/Introduction
```

Single aggregated JSON for the folder:

```bash
python mwe-extraction/extract_mwes_from_conllu_folder.py \
  --input_dir /path/to/conllu/Introduction \
  --output_dir /path/to/ud_mwes_json/Introduction \
  --aggregate
```

### USAS-based MWEs (TXT → JSON)

This extracts **only MWEs** detected by PyMUSAS from each input `.txt`.

Per-file JSON (default):

```bash
python mwe-extraction/pymusas_extract_mwes_txt_folder.py \
  --input_dir /path/to/output_imrad_txt/Introduction \
  --output_dir /path/to/usas_mwes_json/Introduction \
  --use_gpu
```

Single aggregated JSON for the folder:

```bash
python mwe-extraction/pymusas_extract_mwes_txt_folder.py \
  --input_dir /path/to/output_imrad_txt/Introduction \
  --output_dir /path/to/usas_mwes_json/Introduction \
  --aggregate \
  --agg_name all_usas_mwes_introduction.json \
  --use_gpu
```

---

## Dispersion analysis

Dispersion can be computed once MWEs are extracted. The paper reports:

- **Document Frequency (DF)** and **DF%**
- **Gries’ DP**, quantifying deviation from an equal-share baseline:

```math
DP = \frac{1}{2}\sum_{i=1}^{N}\left|p_i - s_i\right|
```

Where _pᵢ_ is the observed proportion of an MWE’s occurrences in document _i_, and _sᵢ_ is the expected proportion under the baseline (operationalized as the document’s share of tokens in the section).

---

## How to cite this work

```bibtex
@inproceedings{
giraud2026beyond,
title={Beyond Single Words: {MWE} Identification in Bioinformatics Research Articles and Dispersion Profiling Across {IMR}aD},
author={Giraud, Jurgi and Gargett, Andrew},
booktitle={22nd Workshop on Multiword Expressions (MWE 2026) @EACL2026},
year={2026},
url={https://openreview.net/forum?id=BHg9nM9DlC}
}
```
