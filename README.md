# BulkNMFGenerator

NMF-based synthetic bulk RNA-seq generator with a non-private variant (**NMF**) and a
differentially private variant (**DP-NMF**). This code produced the NMF and DP-NMF
synthetic datasets for TCGA-BRCA and TCGA-COMBINED benchmarked in the accompanying
manuscript (Genome Biology, under revision; provisional title *Benchmarking generative
models for privacy-preserving synthetic bulk RNA-seq data generation*).

It adapts the single-cell generator in
[SingleCellNMFGenerator](https://github.com/AndrewJWicks/SingleCellNMFGenerator) to
continuous, variance-stabilised bulk expression and plugs into the Track I pipeline of the
[CAMDA Health Privacy Challenge starter kit](https://github.com/PMBio/Health-Privacy-Challenge).

## Method

For each training fold:

1. **NMF.** `MiniBatchNMF` (rank 20) on the VST expression matrix gives low-dimensional
   sample embeddings.
2. **Clustering.** k-means (20 clusters) on the embeddings.
3. **Cluster summaries.** The mean expression vector of each cluster.
4. **Noise (DP-NMF only).** Gaussian noise is added to each cluster mean vector with
   standard deviation σ_c = (R / n_c) · √(2 ln(1.25/δ)) / ε, where R is the largest
   per-sample L2 norm in the dataset, n_c is the cluster size and δ = 1e-5. The
   non-private NMF variant skips this step.
5. **Sampling.** A cluster is drawn with probability proportional to its size, and a
   profile is drawn from Normal(cluster mean, σ_global), where σ_global is the pooled
   within-cluster standard deviation, truncated at zero.
6. **Labels.** A random forest trained on the real embeddings and labels assigns a label
   to each synthetic profile.

The NMF factorisation, k-means partition, cluster proportions, σ_global and the random
forest are fitted on the training data without added noise. The formal privacy statement
for DP-NMF is given in the manuscript.

## Repository layout

```
config.yaml                  template configuration (the drivers overwrite and restore it)
environment.yaml             conda environment
run_dpnmf.py                 DP-NMF: eps in {1, 4, 10, 50} x 5 folds x 2 datasets
run_nmf.py                   non-private NMF: 5 folds x 2 datasets
src/generators/
  blue_team.py               starter kit CLI with resource profiling
  models/nmf_sampler.py      the NMF / DP-NMF generator
  models/base.py             generator base class (starter kit)
  models/multivariate.py     starter kit baseline, imported by blue_team.py
  utils/prepare_data.py      data loading and splits (starter kit)
  utils/resource.py          runtime, memory and CPU profiling
data/                        input data, not included (see data/README.md)
data_splits/split_indices/   seed-42 five-fold splits used in the benchmark
```

## Installation

```bash
conda env create -f environment.yaml
conda activate health-privacy-env
```

Tested on macOS (x86_64, CPU only) with Python 3.10 and scikit-learn 1.5.2.

## Data

The preprocessed TCGA-BRCA and TCGA-COMBINED matrices are distributed by the challenge
organisers and are not included. See [data/README.md](data/README.md) for how to obtain
them and where to place them.

## Reproducing the benchmark datasets

```bash
python run_dpnmf.py    # DP-NMF
python run_nmf.py      # non-private NMF
```

Both drivers skip runs whose outputs already exist. A single run can be launched through
the starter kit CLI after editing `config.yaml`:

```bash
python src/generators/blue_team.py run-generator 1 --experiment_name my_run
```

Outputs:

```
data_splits/{dataset}/synthetic/nmf_sampler/{experiment}/synthetic_data_split_{1..5}.csv     samples x 978 genes
data_splits/{dataset}/synthetic/nmf_sampler/{experiment}/synthetic_labels_split_{1..5}.csv   one column, "Subtype"
resource_logs/{experiment}_split{n}.json                                                      runtime, memory, CPU
```

Labels are molecular subtypes for TCGA-BRCA (e.g. `BRCA.LumA`) and TCGA project codes for
TCGA-COMBINED (e.g. `TCGA-LUAD`, `TCGA-LUSC`). All randomness is seeded (seed 42).

## Attribution and license

`base.py`, `prepare_data.py`, `multivariate.py` and the `blue_team.py` CLI derive from the
[Health Privacy Challenge starter kit](https://github.com/PMBio/Health-Privacy-Challenge)
(GPL-3.0); `prepare_data.py` was modified to select the label column by name. The resource
profiler was provided by the benchmark organisers and extended with CPU reporting. This
repository is released under the GNU General Public License v3.0, see [LICENSE](LICENSE).

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff). Please also cite the manuscript once
published.
