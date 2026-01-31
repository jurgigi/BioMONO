# BioMONO_en: A corpus of bioinformatics research articles

This repository provides the codebase used to (i) build a corpus of bioinformatics research articles and (ii) extract multiword expressions (MWEs) from that corpus. The corpus itself is distributed separately (see **Corpus access** below). This repo focuses on transparent, reproducible data processing and extraction pipelines.

## Corpus access

The corpus is available here: **[LINK TO CORPUS]**  
- Landing page / record: https://doi.org/10.6084/m9.figshare.31215955 

> **Note:** If the corpus includes full-text articles, please ensure redistribution complies with publisher terms and applicable licensing. If needed, distribute only identifiers/metadata and processing scripts, or a derived representation permitted by the source licenses.

## Repository structure

```text
.
├── scripts_corpus_building/     # Python scripts to collect, filter, and structure the corpus
├── scripts_mwe_extraction/      # Python scripts to extract MWEs (multiple pipelines if applicable)
├── LICENSE                      # Code license
└── README.md
