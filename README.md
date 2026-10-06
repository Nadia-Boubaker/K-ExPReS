# K-ExPReS

**K-ExPReS** (*Knowledge-based Explainable and Personalized Recommender System*) is a neuro-symbolic framework that combines deep-learning recommendation with symbolic knowledge representation. It produces personalized recommendations together with post-hoc explanations derived from a Knowledge Graph (KG).

This repository accompanies the paper:

> N. Ben Hadj Boubaker, N. Yacoubi Ayadi, Z. Kodia. **K-ExPReS: A Knowledge Graph-based Explainable and Personalized Recommender System with User-Centered Evaluation.** [journal / venue to be completed]

K-ExPReS extends our earlier work on KA-ERN (MEDES 2024) and path-based explanations for course recommendation (CoopIS 2025).

## Overview

The framework has four components:

1. **KG construction**: users, items and domain entities are connected in a User-Item KG (UIKG).
2. **KG embedding**: entities and relations are embedded with **TransR** (dimension 100).
3. **Recommendation module**: the user and item embeddings are treated as a two-step sequence and passed through a **Bi-LSTM**, an **attention layer** and fully connected layers, which output the interaction probability.
4. **Explainability module** (post-hoc, independent of the recommender): **meta-path guided extraction** of user-to-item paths in the KG, followed by **path ranking** on three criteria (simplicity, popularity, diversity).

## Scope of this repository

| Part of the paper | Included here |
|---|---|
| Recommendation module on **MI** (MovieLens-1M + IMDb) | Yes |
| Explainability module (path extraction and ranking) | Yes, runs on the **EduKA** KG (see below) |
| User study web application | No (live app linked below) |
| KKBox experiments | No. KKBox results in the paper were obtained with the same pipeline |

## Repository structure

```
K-ExPReS/
├── data/
│   ├── train_ratings.csv        # MI interaction splits
│   ├── val_ratings.csv
│   ├── test_ratings.csv
│   ├── movies.csv               # movie metadata
│   └── triplets_train.csv       # training KG (head, relation, tail)
├── embeddings/
│   ├── entity_embeddings_movie_train.npy     # pretrained TransR entity embeddings
│   ├── relation_embeddings_movie_train.npy   # pretrained TransR relation embeddings
│   ├── movie_entity_mapping.csv              # row index <-> entity name
│   └── movie_relation_mapping.csv            # row index <-> relation name
├── src/
│   ├── TransR.py                # trains the KG embeddings
│   ├── Model.py                 # trains and evaluates the recommender
│   └── Explainability/
│       ├── ExtractInformation.py
│       ├── ExtractPaths.py
│       └── PathRanking.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Python 3.11 is recommended (the pinned versions of `pandas` and `numpy` do not support Python 3.12 and later).

```bash
git clone https://github.com/Nadia-Boubaker/K-ExPReS.git
cd K-ExPReS
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A GPU is recommended for training the recommender. The installation above provides the CPU build of PyTorch on Windows.

## Data

### MI (recommendation experiments)

MI merges **MovieLens-1M** (user-item interactions) with **IMDb** (genres, actors, directors, writers), linked through film titles and release dates.

- `triplets_train.csv` contains the KG built from the **training interactions only**: no validation or test interaction appears in the graph, which avoids information leakage.
- Entities are prefixed by type: `U_` (user), `I_` (movie), `G_` (genre), `Ac_` (actor), `D_` (director).
- `embeddings/*_mapping.csv` give the correspondence between embedding rows and entity or relation names.

MovieLens-1M: https://grouplens.org/datasets/movielens/

### EduKA (explainability experiments)

The explainability module and the user study use **EduKA**, a User-Item KG built from a Coursera dataset (1,000 users, 3,533 courses, 68,801 interactions). EduKA is not stored in this repository and is available on Zenodo:

https://zenodo.org/records/15804872

Download it and place it in `data/` before running the explainability scripts. [file name / expected path to be completed]

## Usage

Run all commands from the repository root. Input and output paths are defined at the top of each script.

### 1. Train the TransR embeddings (optional)

Pretrained embeddings are provided in `embeddings/`. To retrain them:

```bash
python src/TransR.py
```

### 2. Train and evaluate the recommender

```bash
python src/Model.py
```

The script negative-samples the splits, trains the Bi-LSTM + attention model, and reports Precision, Recall, F1 and AUC on the validation and test sets.

Settings used in the paper:

| Parameter | Value |
|---|---|
| Embedding dimension | 100 |
| Bi-LSTM hidden units | 256 |
| Fully connected layers | 128, 64, then 1 output unit |
| Dropout | 0.3 |
| Optimizer | Adam, learning rate 0.001, weight decay 1e-5 |
| Gradient clipping | max norm 5.0 |
| Batch size / epochs | 256 / 100 (fixed, no early stopping) |
| Loss | Binary cross-entropy with logits |
| Negative sampling | 1 uniform negative per positive, excluding every item the user has seen in train, validation or test |
| Classification threshold | 0.4 (AUC is threshold-independent) |
| Random seed | 42 |

### 3. Generate explanations

```bash
python src/Explainability/ExtractInformation.py   # [role of this script]
python src/Explainability/ExtractPaths.py         # [role of this script]
python src/Explainability/PathRanking.py          # [role of this script]
```

[Order of execution, inputs and outputs of each script to be completed.]

## Results

### Recommendation performance (Table 3 of the paper)

| Dataset | Model | P | R | F1 | AUC |
|---|---|---|---|---|---|
| KKBox | RippleNet | 0.699 | 0.732 | 0.715 | 0.762 |
| | MEIRec | 0.753 | 0.774 | 0.763 | 0.819 |
| | KPRN | 0.805 | 0.822 | 0.813 | 0.834 |
| | PeRN | **0.842** | 0.861 | 0.851 | 0.866 |
| | **K-ExPReS** | 0.817 | **0.877** | **0.859** | **0.892** |
| MI | RippleNet | 0.742 | 0.713 | 0.727 | 0.694 |
| | MEIRec | 0.792 | 0.804 | 0.798 | 0.734 |
| | KPRN | **0.843** | 0.826 | 0.834 | 0.812 |
| | PeRN | 0.835 | 0.871 | 0.853 | 0.851 |
| | **K-ExPReS** | 0.831 | **0.884** | **0.857** | **0.929** |

K-ExPReS obtains the best Recall, F1 and AUC on both datasets, and stays competitive on Precision.

### Explainability

The ranking function scores each candidate path as

```
Score(p) = α · Simplicity(p) + β · Popularity(p) + γ · Diversity(p)
```

with α = 0.4, β = 0.3, γ = 0.3, where:

- **Simplicity** favours short paths: `1 / (|p| - 1)`.
- **Popularity** is the mean PageRank of the nodes in the path.
- **Diversity** is `1 - max Jaccard` between the path and the paths already selected.

Paths are extracted with four meta-paths over EduKA (U: user, C: course, T: concept):

| ID | Meta-path |
|---|---|
| P1 | U -hasInterest→ C -hasKnowledgeTopic→ T -isKnowledgeTopicOf→ C |
| P2 | U -hasInterest→ C -hasKnowledgeTopic→ T -contributesTo→ T -isKnowledgeTopicOf→ C |
| P3 | U -hasInterest→ C -hasKnowledgeTopic→ T -superTopicOf→ T -isKnowledgeTopicOf→ C |
| P4 | U -hasInterest→ C -hasKnowledgeTopic→ T -superTopicOf⁻¹→ T -superTopicOf→ T -isKnowledgeTopicOf→ C |

### User study

20 participants evaluated the explanations through a web application: https://pathbased.pythonanywhere.com/

| Dimension | Mean (1-5) |
|---|---|
| Explanation clarity | 4.20 |
| Understanding of relationships | 4.15 |
| Alignment with learning goals | 3.70 |
| Trust in the recommender | 3.35 |

70% of participants preferred the path-based explanations over popularity-based and collaborative-filtering-based ones. The application code and the participants' responses are not included in this repository.

## Reproducibility

- All random seeds (Python, NumPy, PyTorch, CUDA) are set to 42.
- LSTM training on GPU is not fully deterministic even with fixed seeds, so results can vary slightly between runs.
- [Hardware and library versions used for the reported results to be completed.]

## Citation

```bibtex
@article{benhadjboubaker2026kexpres,
  title   = {K-ExPReS: A Knowledge Graph-based Explainable and Personalized Recommender System with User-Centered Evaluation},
  author  = {Ben Hadj Boubaker, Nadia and Yacoubi Ayadi, Nadia and Kodia, Zahra},
  journal = {[to be completed]},
  year    = {2026}
}
```

## Contact

Nadia Ben Hadj Boubaker: nadia.ben-hadj-boubaker@etu.univ-lyon1.fr

SMART-LAB, University of Tunis, and LIRIS, Université Claude Bernard Lyon 1.

## License

Released under the [MIT License](LICENSE).
