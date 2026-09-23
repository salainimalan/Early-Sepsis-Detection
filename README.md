# Early Sepsis Detection Using Sequential Pattern Mining and Temporal Classification

An end-to-end sequential data mining pipeline for early sepsis detection from hourly ICU observations using the PhysioNet/CinC 2019 Sepsis Prediction Challenge Training Set A.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Temporal%20Model-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Evaluation-F7931E?logo=scikit-learn&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-2E8B57)
![Task](https://img.shields.io/badge/Task-Sepsis%20Prediction-8A2BE2)
![Method](https://img.shields.io/badge/Method-Sequential%20Mining-0077B6)

**Topics:** `sepsis-prediction` `sequential-pattern-mining` `time-series` `icu-data` `healthcare-ai` `prefixspan` `lstm` `bidirectional-lstm` `sequence-similarity` `jaccard-similarity` `machine-learning` `deep-learning` `data-mining` `physionet`

## Overview

This project transforms continuous ICU measurements into symbolic physiological states, discovers recurring patterns in pre-sepsis and non-septic cohorts, filters those patterns into closed and discriminative patterns, converts them into sequence-similarity features, and performs temporal sepsis classification using a Bidirectional LSTM.

```text
Raw ICU Time Series
        ↓
Exploratory Data Analysis
        ↓
Preprocessing & Symbolic State Generation
        ↓
Pre-Sepsis / Negative Sequence Windows
        ↓
Frequent Sequential Pattern Mining
        ↓
Closed Pattern Filtering
        ↓
Discriminative Pattern Selection
        ↓
Sequence Similarity & Distance Features
        ↓
Bidirectional LSTM Temporal Classification
        ↓
Sepsis Prediction & Evaluation
```

## Dataset

The project uses Training Set A from the PhysioNet/CinC 2019 Sepsis Prediction Challenge.

Analysed cohort:

| Statistic | Value |
|---|---:|
| Patients | 20,336 |
| Non-septic | 18,546 |
| Septic | 1,790 |
| Observation frequency | Hourly |

The raw dataset is not included in this repository.

## Methodology

### 1. Exploratory Analysis

The dataset is examined for patient-level sequence lengths, sepsis-label onset, class imbalance, missing observations, clinical-variable distributions, correlations and patient trajectories.

### 2. Symbolic Sequence Generation

Selected continuous clinical variables are converted into symbolic states using percentile-based thresholds:

```text
LOW / NORMAL / HIGH
```

An hourly observation can therefore contain states such as:

```text
HR_HIGH
MAP_HIGH
O2Sat_NORMAL
Resp_HIGH
SBP_HIGH
```

Pre-sepsis windows are constructed around the observed first positive sepsis-label hour for septic patients, while comparable windows are selected from non-septic patients.

### 3. Frequent Pattern Mining

Positive and negative cohorts are mined independently using PrefixSpan.

```text
Minimum support = 0.01
Maximum pattern length = 5
```

The strongest displayed positive-cohort pattern has support of approximately `0.039`, while the strongest displayed negative-cohort pattern has support of approximately `0.019`.

Example positive pattern:

```text
DBP_HIGH + HR_HIGH + MAP_HIGH +
O2Sat_NORMAL + Resp_HIGH + SBP_HIGH
```

### 4. Closed Pattern Filtering

Frequent mining can produce overlapping patterns containing redundant information. Closed-pattern filtering retains patterns for which no longer pattern contains the same pattern while maintaining identical support.

This reduces redundancy while preserving representative support information.

### 5. Discriminative Pattern Selection

Closed patterns are compared between the pre-sepsis and negative cohorts.

A pattern is retained when:

```text
Positive support ≥ 0.01
Support difference ≥ 0.01
Positive-to-negative support ratio ≥ 1.0
```

Five patterns satisfy the final criteria.

Their observed ranges are:

```text
Positive support:    0.0146 – 0.0387
Support difference:  0.0101 – 0.0282
```

The strongest selected pattern has positive support `0.0387` and support difference `0.0282`.

### 6. Sequence Similarity and Distance

Each selected pattern is compared with hourly itemsets using Jaccard-style similarity:

```text
Similarity(A,B) = |A ∩ B| / |A ∪ B|
Distance(A,B) = 1 - Similarity(A,B)
```

For every patient-pattern pair, minimum distance, maximum similarity, best matching position and mean similarity are derived.

The resulting feature matrix contains:

```text
19,855 instances
20 similarity/distance features
```

Average maximum similarity:

```text
Positive ≈ 0.52
Negative ≈ 0.45
```

Average mean similarity:

```text
Positive ≈ 0.35
Negative ≈ 0.30
```

### 7. Temporal Classification

A 10-hour historical ICU window is used to predict the sepsis state in the following hour.

The temporal branch preserves the hourly order of:

```text
HR, MAP, O2Sat, SBP, Resp
```

Additional clinical variables are represented through window-level summaries.

A stacked Bidirectional LSTM learns temporal dependencies, combines the learned representation with additional clinical features, and produces the final two-class prediction. Class weighting is used to address the strong class imbalance.

## Final Validation Results

| Metric | Result |
|---|---:|
| **AUROC** | **0.79196** |
| **AUPRC** | **0.07650** |
| **Accuracy** | **0.84982** |
| **Precision** | **0.07221** |
| **Recall** | **0.48745** |
| **F1-score** | **0.12578** |

Because septic cases are a minority class, accuracy should not be interpreted independently. AUPRC, precision, recall and F1 provide additional information about minority-class detection.

## Evaluation Plots

The final model generates:

- Validation ROC curve
- Validation Precision–Recall curve
- AUROC and AUPRC values
- Classification metrics

Recommended figure locations:

```text
figures/validation_roc.png
figures/validation_pr.png
```

## Repository Structure

```text
.
├── data/
│   └── training/
│       └── *.psv
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_Preprocessing_and_Sequence_Building.ipynb
│   ├── 02_Sequential_Pattern_Mining.ipynb
│   ├── 03_Closed_and_Discriminative_Patterns.ipynb
│   ├── 04_Sequence_Distance_Features.ipynb
│   └── 05_Temporal_Classification.ipynb
├── outputs/
├── figures/
├── requirements.txt
├── README.md
└── LICENSE
```

Adjust filenames if the repository uses a different final organization.

## Installation

Recommended environment:

- Python 3.9+
- Jupyter Notebook/JupyterLab
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- PyTorch
- TensorFlow/Keras where required by the final model
- PrefixSpan-compatible implementation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place the required Training Set A PSV files under:

```text
data/training/
```

The raw dataset should not be committed to the repository because of its size and dataset-access requirements.

## Running the Project

Run the notebooks in order:

```text
00_EDA
→ 01_Preprocessing_and_Sequence_Building
→ 02_Sequential_Pattern_Mining
→ 03_Closed_and_Discriminative_Patterns
→ 04_Sequence_Distance_Features
→ 05_Temporal_Classification
```

Each stage generates intermediate outputs used by subsequent stages.

## Why This Approach?

The project combines three complementary components:

**Sequential pattern mining** discovers recurring physiological configurations associated with the pre-sepsis period.

**Pattern-based similarity** measures how closely a patient's recent physiological states resemble the discovered patterns.

**Temporal deep learning** learns dependencies across ordered ICU observations and performs the final prediction.

This produces a pipeline that combines interpretable physiological pattern discovery with temporal predictive modeling.

## Limitations

- Only Training Set A is used because of computational constraints.
- The dataset has substantial class imbalance.
- Symbolic discretization reduces continuous numerical information.
- Pattern discovery depends on support and discriminative thresholds.
- Negative-window sampling may not represent every challenging non-septic case.
- The system is an academic research prototype and is not intended for clinical deployment.

## Visual Results

### ICU Data Overview

![ICU data overview](figures/eda_overview.png)

### Frequent Pre-Sepsis Patterns

![Frequent positive-cohort patterns](figures/frequent_positive_patterns.png)

### Frequent Negative-Cohort Patterns

![Frequent negative-cohort patterns](figures/frequent_negative_patterns.png)

### Discriminative Patterns

![Discriminative patterns](figures/discriminative_patterns.png)

### Pattern Similarity by Cohort

![Pattern similarity by cohort](figures/pattern_similarity.png)

### Validation ROC Curve

![Validation ROC curve](figures/validation_roc.png)

## Academic Context

**Course:** CSE 3068 – Sequential and Spatial Data Mining

**Project:** Early Sepsis Detection Using Sequential Pattern Mining and Distance-Based Classification on ICU Vital-Sign Sequences

## Disclaimer

This project is intended for academic and research purposes only. It is not a clinical diagnostic system and should not be used for medical decision-making.

## Author

**Salai Nimalan**  
Integrated M.Tech. Computer Science and Business Analytics
