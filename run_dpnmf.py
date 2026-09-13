#!/usr/bin/env python3
"""
Driver: generate DP-NMF synthetic datasets for both TCGA datasets across
epsilon in {1, 4, 10, 50}, all five seed-42 folds, through the official
blue_team.py + resource.py harness.

For each (dataset, epsilon) it writes config.yaml and runs all 5 splits.
- synthetic CSVs -> data_splits/{dataset}/synthetic/nmf_sampler/{dataset}_eps{e}/
- resource logs  -> resource_logs/{dataset}_eps{e}_split{n}.json
"""
import os
import sys
import copy
import subprocess
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
EPSILONS = [1, 4, 10, 50]
SPLITS = [1, 2, 3, 4, 5]

DATASETS = {
    "TCGA-BRCA": {
        "count_file": "data/processed/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv",
        "annot_file": "data/meta/TCGA-BRCA_primary_tumor_subtypes.csv",
        "subtype_col_name": "Subtype",
    },
    "TCGA-COMBINED": {
        "count_file": "data/processed/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv",
        "annot_file": "data/meta/TCGA-COMBINED_primary_tumor_subtypes.csv",
        # project CODES (TCGA-LUAD/TCGA-LUSC/TCGA-KIRC/TCGA-KIRP ...), NOT the lossy
        # tissue text (cancer_type collapses LUAD+LUSC->Lung, KIRC+KIRP->Kidney).
        "subtype_col_name": "project",
    },
}


def public_l2_bound(count_file):
    """Per-dataset L2 contribution bound: max record L2 over the FULL dataset,
    treated as a public (data-independent of any single fold) hyperparameter."""
    X = pd.read_csv(os.path.join(ROOT, count_file), sep="\t", index_col=0).T
    return float(np.linalg.norm(X.values.astype(np.float64), axis=1).max())


def _run():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    for ds, fields in DATASETS.items():
        clip = public_l2_bound(fields["count_file"])
        print(f"\n##### {ds}: public L2 clip bound = {clip:.4f} #####")
        for eps in EPSILONS:
            cfg = copy.deepcopy(base)
            cfg["dataset_config"].update({
                "name": ds,
                "count_file": fields["count_file"],
                "annot_file": fields["annot_file"],
                "subtype_col_name": fields["subtype_col_name"],
            })
            cfg["dp_config"].update({
                "eps": eps, "mechanism": "gaussian", "delta": 1.0e-5, "clip_l2": clip,
            })
            with open(os.path.join(ROOT, "config.yaml"), "w") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)

            exp = f"{ds}_eps{eps}"
            for sp in SPLITS:
                out = os.path.join(ROOT, "data_splits", ds, "synthetic",
                                   "nmf_sampler", exp, f"synthetic_data_split_{sp}.csv")
                if os.path.exists(out):
                    print(f"--- {ds} eps={eps} split={sp} : already done, skipping ---")
                    continue
                print(f"--- {ds} eps={eps} split={sp} ---")
                subprocess.run(
                    [sys.executable, "src/generators/blue_team.py",
                     "run-generator", str(sp), "--experiment_name", exp],
                    cwd=ROOT, check=True)
    print("\nAll DP-NMF runs complete.")


def main():
    # blue_team.py reads ./config.yaml, so each run rewrites it.
    # Restore the original afterwards to keep the working tree clean.
    cfg_path = os.path.join(ROOT, "config.yaml")
    with open(cfg_path) as fh:
        original = fh.read()
    try:
        _run()
    finally:
        with open(cfg_path, "w") as fh:
            fh.write(original)


if __name__ == "__main__":
    main()
