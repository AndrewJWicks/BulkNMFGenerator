# Input data

The preprocessed TCGA matrices are distributed through the
[ELSA Benchmarks platform](https://benchmarks.elsa-ai.eu/?ch=4) (CAMDA Health Privacy
Challenge, Track I; registration required) and are not redistributed here.
Preprocessing (DESeq2 VST, 978 L1000 landmark genes) is described in the
[starter kit data README](https://github.com/PMBio/Health-Privacy-Challenge/blob/main/data/README.md).

Place the files as follows.

```
data/processed/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv       genes x samples, 1089 samples
data/processed/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv   genes x samples, 4323 samples
data/meta/TCGA-BRCA_primary_tumor_subtypes.csv                          includes samplesID, Subtype
data/meta/TCGA-COMBINED_primary_tumor_subtypes.csv                      includes samplesID, project
```

The label column is `Subtype` for TCGA-BRCA and `project` (TCGA project code) for
TCGA-COMBINED.
