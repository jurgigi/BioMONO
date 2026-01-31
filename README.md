# BioMONO_en: A corpus of bioinformatics research articles

This repository provides the codebase used to in the article "Beyond Single Words: MWE Identification in Bioinformatics Research Articles and Dispersion Profiling Across IMRaD" to (i) build a corpus of bioinformatics research articles and to segment it into main IMRaD sections and (ii) to extract Multiword Expressions (MWEs) from that corpus. The corpus itself is distributed separately (see **Corpus access** below).

## Corpus access

The corpus (already segmented into IMRaD sections) is available for download here: 
https://doi.org/10.6084/m9.figshare.31215955 


## Repository structure

```text
.
├── corpus-building/     # Python scripts to collect, filter, and structure the corpus
├── mwe-extraction/      # Python scripts to extract MWEs (UD and USAS)
├── mwes-lists           # lists of MWEs from ACL, ARTES and the MeSH controlled vocabulary thesaurus
├── LICENSE              # Code license
├── README.md
└── requirements.txt
