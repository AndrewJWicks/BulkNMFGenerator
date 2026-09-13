#!/usr/bin/env python3
"""
Bulk DP-NMF synthetic data generator (Track I: TCGA-BRCA / TCGA-COMBINED).

Adapted from the original single-cell `nmf_sampler` concept, keeping the pipeline
intact:
    MiniBatchNMF -> basis H, embeddings W              (non-private, internal)
    -> RandomForest on (W, labels)                     (non-private, internal)
    -> KMeans clustering of W                          (non-private, internal)
    -> per-cluster gene-mean summaries
    -> Gaussian noise on the per-cluster gene means    (DP-NMF only)
    -> sample synthetic profiles per cluster
    -> assign subtype labels via the RandomForest on synthetic embeddings

Two adaptations vs. the single-cell version:
  1. Sampler: single-cell data are integer counts (Poisson, whose variance is a
     function of the mean); TCGA data are *continuous, variance-stabilised* (VST)
     values, so we sample from Normal(mean, sigma_global). Because VST is
     homoscedastic by construction, a single global standard deviation is the
     faithful continuous analogue of "variance determined by the released mean".
  2. Noise is added to the per-cluster MEANS, from which the synthetic profiles
     are drawn. Perturbing only the NMF basis H left the output essentially
     unchanged across epsilon, because profiles are sampled from the cluster
     summaries rather than from H.

Noise mechanism (DP-NMF): each per-cluster mean vector receives Gaussian noise with
sigma_c = (R / n_c) * sqrt(2 ln(1.25/delta)) / eps, where R is a per-sample L2 norm
bound (the drivers use the largest per-sample L2 norm in the dataset) and n_c is
the cluster size. The NMF factorisation, k-means partition, cluster proportions,
sigma_global and the RandomForest labeler are fitted on the training data without
added noise. The formal privacy statement for this model is given in the
accompanying manuscript.

A Laplace variant (L1 sensitivity) is also implemented but is impractically noisy
in ~10^3 gene dimensions; Gaussian is the default.
"""

import os
import sys
import random
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import MiniBatchNMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from generators.models.base import BaseDataGenerator


class NMFSamplerDataGenerator(BaseDataGenerator):
    def __init__(self, config: Dict[str, Any], split_no: int = 1):
        super().__init__(config, split_no)

        nmf_cfg = self.generator_config  # == config["nmf_sampler_config"]
        self.dp_cfg = config.get("dp_config", {})

        self.seed          = int(nmf_cfg.get("seed", 42))
        self.n_components  = int(nmf_cfg.get("n_components", 20))
        self.nmf_batch     = int(nmf_cfg.get("nmf_batch_size", 1000))
        self.n_clusters_cfg = int(nmf_cfg.get("n_clusters", 20))
        self.requested     = int(nmf_cfg.get("n_synth_samples", -1))
        self.prop_aware    = bool(nmf_cfg.get("proportion_aware", True))

        # fitted state (populated in train())
        self.nmf = None
        self.rf = None
        self.classes_ = None
        self.gene_means = None       # DP-noised per-cluster gene means
        self.sigma_global = None     # scalar sampling SD (VST is homoscedastic)
        self.cluster_probs = None
        self.n_clusters = None
        self.feature_names = None
        self.n_train = None

        random.seed(self.seed)
        np.random.seed(self.seed)

    # ------------------------------------------------------------------ I/O
    def _load_train(self):
        """Load this fold's training rows DIRECTLY from the provided
        {dataset}_splits.yaml + raw count/annotation files.

        No split (re)generation or materialization: the seed-42 fold definitions
        in splits.yaml are used verbatim; we only index the raw data by the
        train sample IDs for `self.split_no`.
        """
        import yaml

        home   = self.config["dir_list"]["home"]
        dcfg   = self.dataset_config
        sample_col  = dcfg["sample_col_name"]
        subtype_col = dcfg["subtype_col_name"]

        # raw expression: genes x samples -> samples x genes
        X = pd.read_csv(os.path.join(home, dcfg["count_file"]),
                        sep="\t", index_col=0).T
        # raw annotations, aligned to the expression sample order
        ann = pd.read_csv(os.path.join(home, dcfg["annot_file"]))
        ann = ann.set_index(sample_col).loc[X.index]
        labels = ann[subtype_col].astype(str)

        # provided fold definition (seed-42), used as-is
        splits_path = os.path.join(home, self.config["dir_list"]["split_save_dir"],
                                   f"{self.dataset_name}_splits.yaml")
        with open(splits_path) as fh:
            splits = yaml.safe_load(fh)["splits"]
        train_ids = splits[f"split_{self.split_no}"]["train_index"]

        X_tr = X.loc[train_ids]
        y_tr = labels.loc[train_ids]
        self.feature_names = X.columns.values
        return X_tr.values.astype(np.float64), y_tr.values

    # ------------------------------------------------------------------ DP
    def _dp_noise_means(self, gene_means_clean, cluster_sizes, V):
        """Gaussian (default) or Laplace noise on the per-cluster gene means.

        Returns the noised means clipped at zero, or the clean means when eps is
        unset (non-private NMF).
        """
        eps = self.dp_cfg.get("eps")
        if eps is None or float(eps) <= 0:
            print("[INFO] No DP noise on cluster means (non-private baseline).")
            return gene_means_clean

        eps = float(eps)
        mech = str(self.dp_cfg.get("mechanism", "gaussian")).lower()

        # Per-sample L2 norm bound R. The drivers pass the largest per-sample L2 norm
        # of the dataset via `clip_l2`; if unset, fall back to the largest norm in
        # the training fold.
        R = self.dp_cfg.get("clip_l2")
        if R is None:
            R = float(np.linalg.norm(V, axis=1).max())
            print(f"[WARN] clip_l2 not set; using the largest per-sample L2 norm in "
                  f"the training fold ({R:.4g}) as R.")
        else:
            R = float(R)

        noised = gene_means_clean.copy()
        g = gene_means_clean.shape[1]
        if mech == "laplace":
            # pure eps-DP via L1 over the full gene vector (very noisy in high-dim)
            c = float(V.max())
            for cid in range(self.n_clusters):
                nc = max(1, int(cluster_sizes[cid]))
                scale = (g * c / nc) / eps
                noised[cid] = gene_means_clean[cid] + np.random.laplace(0, scale, size=g)
            print(f"[INFO] Noise on cluster means: Laplace, eps={eps}, "
                  f"per-gene bound c={c:.4g}")
        else:
            delta = float(self.dp_cfg.get("delta", 1e-5))
            k = np.sqrt(2.0 * np.log(1.25 / delta))
            for cid in range(self.n_clusters):
                nc = max(1, int(cluster_sizes[cid]))
                sigma = (R / nc) * k / eps
                noised[cid] = gene_means_clean[cid] + np.random.normal(0, sigma, size=g)
            print(f"[INFO] Noise on cluster means: Gaussian, eps={eps}, "
                  f"delta={delta}, R={R:.4g}")
        return np.clip(noised, 0.0, None)

    # ------------------------------------------------------------------ fit
    def train(self):
        V, y = self._load_train()
        self.n_train = V.shape[0]
        n_samples, n_genes = V.shape
        bs = max(1, min(self.nmf_batch, n_samples))

        # 1) NMF (internal, non-private) -> latent embeddings W
        self.nmf = MiniBatchNMF(n_components=self.n_components,
                                random_state=self.seed)
        for i in range(0, n_samples, bs):
            self.nmf.partial_fit(V[i:i + bs])
        W = self.nmf.transform(V)

        # 2) RandomForest for label assignment (internal, non-private)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        self.rf.fit(W, y)
        self.classes_ = self.rf.classes_

        # 3) KMeans clustering of the latent space (internal, non-private)
        self.n_clusters = int(min(max(2, self.n_clusters_cfg), n_samples))
        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        labels = km.fit_predict(W)
        cluster_sizes = np.bincount(labels, minlength=self.n_clusters)

        # 4) Per-cluster gene-mean summaries
        gene_means = np.zeros((self.n_clusters, n_genes))
        within_var = []
        for cid in range(self.n_clusters):
            mask = labels == cid
            if mask.sum() > 0:
                sub = V[mask]
                gene_means[cid] = sub.mean(axis=0)
                within_var.append(sub.var(axis=0))
        # single global SD (VST is homoscedastic -> faithful continuous analogue
        # of Poisson's mean-determined variance; not a per-gene released summary)
        self.sigma_global = float(np.sqrt(np.mean(within_var))) if within_var else 1.0

        # 5) Noise on the cluster means (DP-NMF only)
        self.gene_means = self._dp_noise_means(gene_means, cluster_sizes, V)

        if self.prop_aware and cluster_sizes.sum() > 0:
            self.cluster_probs = cluster_sizes / cluster_sizes.sum()
        else:
            self.cluster_probs = np.ones(self.n_clusters) / self.n_clusters

        print(f"[INFO] Trained DP-NMF on {n_samples} samples x {n_genes} genes "
              f"(K={self.n_components}, clusters={self.n_clusters}, "
              f"sigma_global={self.sigma_global:.3f}). "
              f"Cluster sizes: {cluster_sizes.tolist()}")

    # -------------------------------------------------------------- generate
    def generate(self):
        if self.nmf is None:
            self.train()

        n_synth = self.requested if self.requested > 0 else self.n_train
        n_genes = self.gene_means.shape[1]

        cids = np.random.choice(self.n_clusters, size=n_synth, p=self.cluster_probs)
        synth = np.empty((n_synth, n_genes), dtype=np.float64)
        for j, cid in enumerate(cids):
            synth[j] = np.random.normal(self.gene_means[cid], self.sigma_global)
        synth = np.clip(synth, 0.0, None)  # respect the non-negative VST domain

        # assign subtype labels via the RandomForest on synthetic embeddings
        synth_W = self.nmf.transform(synth)
        proba = self.rf.predict_proba(synth_W)
        labels = np.array([
            self.classes_[np.random.choice(len(self.classes_), p=proba[i])]
            for i in range(n_synth)
        ])

        features = pd.DataFrame(synth, columns=self.feature_names)
        labels = pd.Series(labels, name="Subtype")
        return features, labels

    def generate_for_type(self, subtype: str = None, num_samples: int = None):
        """Generate samples whose assigned label == `subtype`."""
        if self.nmf is None:
            self.train()
        if num_samples is None:
            num_samples = self.n_train
        feats, lbls = [], []
        while sum(len(f) for f in feats) < num_samples:
            f, l = self.generate()
            m = (l.values == subtype)
            if m.any():
                feats.append(f[m])
                lbls.append(l[m])
        features = pd.concat(feats, ignore_index=True).iloc[:num_samples]
        labels = pd.concat(lbls, ignore_index=True).iloc[:num_samples]
        return features, labels

    def load_from_checkpoint(self):
        # Stateless across runs: (re)fit in train(). Present for interface parity.
        self.train()
