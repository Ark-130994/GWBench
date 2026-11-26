
<div align="center">

# Uncovering Challenges of Solving the Continuous Gromov-Wasserstein Problem

[Xavier Aramayo](https://scholar.google.com/citations?user=1B9UIYoAAAAJ), [Maksim Nekrashevich](https://scholar.google.com/citations?user=bhHV5dQAAAAJ), [Petr Mokrov](https://scholar.google.cat/citations?user=CRsi4IkAAAAJ), 
[Evgeny Burnaev](https://scholar.google.com/citations?user=pCRdcOwAAAAJ), [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ)


[![arXiv Paper](https://img.shields.io/badge/arXiv-2502.01416-b31b1b)](https://arxiv.org/abs/2303.05978) [![OpenReview Paper](https://img.shields.io/badge/OpenReview-PDF-8c1b13)](https://openreview.net/forum?id=GVX9tTLcIs) ![GitHub License](https://img.shields.io/github/license/gregkseno/csbm) 
[![GitHub Stars](https://img.shields.io/github/stars/Ark-130994/GWBench?style=social)](https://github.com/Ark-130994/GWBench/stargazers)
 [![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/XavierArCa/GWBench/tree/main)

</div>

This repository contains the official implementation of the paper "Uncovering Challenges of Solving the Continuous Gromov-Wasserstein Problem", accepted at NeurIPS 2026.

##  1. 💾 Preprocessing of Embeddings


We use three datasets with different embeddings and source-to-target dimensions:

| Dataset | Embedding | Source → Target dimensions |
|---|---|---|
| Twitter | [GloVe](https://github.com/piskvorky/gensim) | 100 → 50 |
| [MUSE](https://github.com/facebookresearch/MUSE) | [Byte-Pair Embeddings (BPEmb)](https://github.com/bheinzerling/bpemb) | 100 → 50 |
| [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) | [WNTA](https://huggingface.co/Orange/Speaker-wavLM-pro) | 128 → 64 |

Once the datasets are downloaded, embeddings can be obtained using `src/preprocess.py` by specifying the dataset, embedding type, and dimension (and `--vocab-size` for BPEmb). For example:

```bash
python preprocess.py --dataset twitter --embedding glove --dim 100
```

We also provide direct downloable datasets on [HugginFace](https://huggingface.co/datasets/XavierArCa/GWBench/tree/main).

##  2. 🏋️ Running Experiments

In the paper we consider two different type of solvers: baseline and large-scale solvers. Baseline approaches are discrete GW solutions that allow out-of-sample inference by some specific mechanism. On the other hand, large-scale solvers are optimized using minibatch learning and stochastic gradient descent. Below we provide instructions on how to reproduce the results of the paper for both cases.

### 2.1. Baseline (discrete) Solvers. 

For simplicity, we refer to three previously proposed approaches as *AlignGW*, *FlowGW*, and *StructuredGW* throughout the paper. This repository includes YAML configuration files in `configs/discrete` that already contain the selected parameters needed to reproduce our results. We provide scripts to run each method.

To run a toy experiment:

```bash
python scripts/run_baselines_discrete.py --config=configs/discrete/toy/<method_name>.yaml
```

For an embedding experiment:

```bash
python scripts/run_baselines_discrete.py --config=configs/discrete/embeddings/<method_name>.yaml
```

> [!NOTE]
> The tag `<method_name>` should be a lowercase of the method name. For example, for FlowGW it will be flowgw. The same applies to large-scale methods.
>

### 2.2. Large-scale (continuous) Solvers.

The continuous baseline pipeline includes *RegGW*, *CycleGW*, and *FlowGW* (minibatch). 

Run a toy experiment:

```bash
python scripts/run_baselines_continuous.py --config=configs/continuous/toy/<method_name>.yaml
```

For an embedding experiment:

```bash
python scripts/run_baselines_continuous.py --config=configs/continuous/embeddings/<method_name>.yaml
```

### 2.3. Neural inner-GW experiments

The proposed methods is *NeuralGW*. Our method includes an early stopping strategy and our proposed initialization with the parameters included in the configs files.

Run a toy experiment:

```bash
python scripts/toy/run_neural_innergw.py --config=configs/neural_innergw/toy/neuralgw.yaml
```

For an embedding experiment:

```bash
python scripts/run_neural_innergw.py --config=configs/neural_innergw/embeddings/neuralgw.yaml
```

## 3. 📈 Expected Output and Results

Each experiment creates a directory under `results/`, organized by experiment type, solver, dataset, statistical relatedness level, and random seed. The resulting directory structure is illustrated below:

```bash
embedding/
└── continuous/
    └── CycleGW/
        └── twitter_glove_100_50/
            └── alpha_1.0_seed_1/
                ├── 000/                           # Numbered experiment run
                │   ├── ckpts/                     # Model checkpoints
                │   ├── plots/                     # Generated plots
                │   ├── config.yaml                # Experiment configuration
                │   ├── losses.csv                 # Training losses
                │   └── metrics_train_direct.csv   # Training metrics
                └── 001/
                    └── ...
```

In general, results follow the directory structure:

`results/<experiment_type>/<solver_type>/<method>/<dataset>/alpha_<alpha>_seed_<seed>/<run_id>/`

Here, `alpha` represents the statistical relatedness level between source and target datasets, while `seed` identifies the random seed. Each numbered directory (e.g., `000`, `001`) contains the outputs of an experimental run.

This organization facilitates reproducibility and comparison across methods, datasets, statistical relatedness levels, and experimental runs.

### 🎓 Citation

If find the paper or the code useuful, you can use the citation below

```bibtex
@inproceedings{
  carrasco2026uncovering,
  title={Uncovering Challenges of Solving the Continuous Gromov-Wasserstein Problem},
  author={Xavier Aramayo Carrasco and Maksim Nekrashevich and Petr Mokrov and Evgeny Burnaev and Alexander Korotin},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=GVX9tTLcIs}
}
```

