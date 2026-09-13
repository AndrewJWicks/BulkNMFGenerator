#!/usr/bin/env python3
"""
Driver: non-private NMF baseline (no DP noise) for both TCGA datasets, all five
seed-42 folds, through the official blue_team.py + resource.py harness. Produces
resource metrics for plain NMF, for comparison with DP-NMF.

- resource logs  -> resource_logs/{dataset}_NMF_split{n}.json   (incl. CPU details)
- synthetic CSVs -> data_splits/{dataset}/synthetic/nmf_sampler/{dataset}_NMF/
"""
import os
import sys
import copy
import subprocess
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
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
        "subtype_col_name": "project",  # TCGA project codes, e.g. TCGA-LUAD
    },
}


def _run():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    for ds, fields in DATASETS.items():
        cfg = copy.deepcopy(base)
        cfg["dataset_config"].update({
            "name": ds,
            "count_file": fields["count_file"],
            "annot_file": fields["annot_file"],
            "subtype_col_name": fields["subtype_col_name"],
        })
        # non-private NMF: turn DP off
        cfg["dp_config"] = {"eps": None, "mechanism": "gaussian",
                            "delta": 1.0e-5, "clip_l2": None}
        with open(os.path.join(ROOT, "config.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)

        exp = f"{ds}_NMF"
        for sp in SPLITS:
            out = os.path.join(ROOT, "data_splits", ds, "synthetic",
                               "nmf_sampler", exp, f"synthetic_data_split_{sp}.csv")
            if os.path.exists(out):
                print(f"--- {ds} NMF split={sp} : already done, skipping ---")
                continue
            print(f"--- {ds} NMF (non-private) split={sp} ---")
            subprocess.run(
                [sys.executable, "src/generators/blue_team.py",
                 "run-generator", str(sp), "--experiment_name", exp],
                cwd=ROOT, check=True)
    print("\nNon-private NMF baseline complete.")


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
