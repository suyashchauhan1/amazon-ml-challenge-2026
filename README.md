# ML Challenge 2026: Business Entity Resolution

A scalable machine-learning pipeline for resolving business entities across multiple noisy data sources. The system combines multi-attribute blocking, text normalization, pairwise similarity features, gradient-boosted classification, threshold calibration, and target-level deduplication.

---

## 1. Overview

Business Entity Resolution is the task of determining which records from different data sources refer to the same real-world business.

For this challenge, **Source 1** acts as the reference entity set, while **Source 2** and **Source 3** contain noisy records that may correspond to those entities.

The main difficulties include:

* Variations in business names
* Missing or corrupted addresses
* Different address formats and token ordering
* Legal suffix variations such as `Pvt Ltd`, `LLC`, `GmbH`, etc.
* Transliteration between scripts
* Punctuation and formatting differences
* Multiple candidate records for the same reference entity
* Large-scale candidate search across millions of records

The solution therefore uses a multi-stage pipeline rather than comparing every Source 1 record against every target record.

### Pipeline

```text
Raw Source Data
      │
      ▼
Normalization
      │
      ▼
Country Partitioning
      │
      ▼
Multi-Attribute Blocking
      │
      ▼
Candidate Pairs
      │
      ▼
Pairwise Feature Extraction
      │
      ▼
Gradient-Boosted Classifier
      │
      ▼
Probability Thresholding
      │
      ▼
Target Deduplication / Consistency
      │
      ▼
Final Matching Results
```

---

# 2. Methodology

## 2.1 Data Characteristics

The matching problem contains several forms of real-world data noise.

### Country separation

Country is used as an early partitioning dimension. Records are processed within country groups instead of generating unrestricted cross-country comparisons.

This substantially reduces the candidate search space and prevents unnecessary comparisons between records belonging to different country partitions.

### Name variation

The same business may appear with:

* Different capitalization
* Punctuation differences
* Legal suffixes
* Concatenated words
* Different token ordering
* Transliteration into another script
* Minor spelling variations

For example:

```text
Moon Boards Pvt Ltd
Moon Boards
MOONBOARDS.COM
```

may refer to the same underlying business.

### Address variation

Addresses may differ through:

* Abbreviations
* Punctuation
* Component ordering
* Missing components
* Additional landmarks
* Different representations of street or road names
* Differences in house/street numbers

Because of this, both lexical and numerical address information are useful.

---

# 3. Text Normalization

Normalization is performed before blocking and feature extraction.

The normalization layer handles:

* Unicode normalization
* Transliteration
* Case normalization
* Punctuation removal
* Legal suffix handling
* Address abbreviation normalization
* Number extraction
* Whitespace and token normalization

The implementation uses `text-unidecode` to convert non-Latin text into a normalized Romanized representation.

Examples of address normalization include mappings such as:

```text
St  → Street
Rd  → Road
```

while business-name normalization removes unnecessary legal and formatting variations.

The goal is not to decide whether two records match during normalization. Instead, normalization creates representations that make candidate generation and similarity measurement more robust.

---

# 4. Candidate Generation

Comparing every Source 1 entity with every target record would be computationally impractical.

The system therefore uses **multi-attribute inverted-index blocking**.

Instead of:

```text
Every Source 1 record
        ×
Every Source 2/3 record
```

the system retrieves a smaller set of potentially relevant candidates using multiple blocking keys.

### Blocking keys

The implementation generates combinations of normalized name and address information, including keys based on:

* Compact normalized business names
* Core business names
* Leading name tokens
* Significant name tokens
* Address numbers
* Street/locality information
* City/state information
* Name + number combinations
* Name + location combinations
* Address token combinations

Multiple blocking keys are combined so that a genuine match does not have to satisfy a single exact representation.

### Frequency control

Extremely common blocking keys can generate very large candidate groups and introduce substantial noise.

The pipeline therefore applies frequency limits to high-frequency keys while retaining more selective keys.

This allows the system to maintain a practical candidate set without performing exhaustive pairwise comparison.

---

# 5. Pairwise Feature Extraction

Each surviving candidate pair is converted into a numerical feature vector.

The features compare the Source 1 entity with its candidate target record.

## Name features

The name representation includes multiple complementary similarity signals, such as:

* Exact normalized-name equality
* Core-name equality
* Normalized edit similarity
* Jaro-Winkler similarity
* Token-sort similarity
* Token-set similarity
* Token Jaccard similarity
* Length-based features
* Prefix agreement
* Character n-gram similarity
* First-token agreement
* First-token conflict

Using several similarity measures is important because different types of corruption are captured by different metrics.

---

## Address features

Address comparison includes signals such as:

* Address availability
* Exact normalized-address equality
* Edit similarity
* Jaro-Winkler similarity
* Token-sort similarity
* Token-set similarity
* Token Jaccard similarity
* Shared numeric tokens
* Numeric agreement
* Numeric conflict

Numeric conflicts are particularly useful because two businesses can have highly similar textual addresses while still occupying different units or buildings.

---

## Cross-field features

The model also receives interactions between name and address information.

Examples include:

* Exact name with missing address
* Name similarity × address similarity
* Combined name/address similarity
* Maximum and minimum similarity
* Source indicator
* Number of shared blocking keys

These features allow the classifier to distinguish cases where strong evidence comes from one field versus cases where both fields independently support a match.

---

# 6. Matching Model

The project contains a model wrapper that supports gradient-boosted tree classifiers.

### Production model

The current inference implementation uses **XGBoost**, with the default production configuration using GPU acceleration through CUDA.

The XGBoost configuration uses:

```text
tree_method = hist
device      = cuda
n_estimators = 500
learning_rate = 0.035
max_depth    = 7
subsample    = 0.85
colsample_bytree = 0.85
```

The model wrapper also contains support for **LightGBM**, allowing the classifier implementation to be changed without restructuring the rest of the pipeline.

This separation keeps the following components independent:

```text
Candidate Generation
        ↓
Feature Extraction
        ↓
Model
        ↓
Thresholding
        ↓
Deduplication
```

---

# 7. Thresholding and Consistency

The classifier produces a probability for each candidate pair.

A candidate is retained only when its probability satisfies the corresponding decision threshold.

The inference pipeline reads the calibrated thresholds from metadata rather than hard-coding the final decision inside the classifier.

Source-specific thresholds can therefore be used for Source 2 and Source 3.

### Target deduplication

A target record should not be assigned to multiple Source 1 entities when the matching constraint requires one-to-one target assignment.

After scoring candidate pairs, the pipeline applies target-level deduplication and retains the highest-confidence assignment when competing Source 1 records nominate the same target.

This provides an additional consistency constraint after the independent pairwise predictions.

---

# 8. Large-Scale Inference

The complete test set is processed in batches rather than loading every candidate pair into memory simultaneously.

The inference implementation uses:

* Country-based partitioning
* Batch feature construction
* NumPy arrays for model input
* SQLite-backed intermediate caching
* GPU-accelerated XGBoost prediction
* Explicit cleanup of temporary objects

The disk-backed cache is used to avoid retaining the complete candidate space in RAM during inference.

The command-line entry point exposes configurable parameters including:

```text
--test-dir
--model-path
--meta-path
--output-dir
--batch-size
--top-k
--validate
```

---

# 9. Validation and Experiments

The project README originally recorded a series of validation experiments comparing different model configurations, blocking strategies, feature sets, and consistency rules.

The reported experiments include:

| Experiment | Model / Configuration | Candidate Strategy   | Main Change                |
| ---------- | --------------------- | -------------------- | -------------------------- |
| EXP-01     | LightGBM baseline     | SuperBlocking Top-15 | Baseline                   |
| EXP-02     | LightGBM              | SuperBlocking Top-15 | Threshold tuning           |
| EXP-03     | LightGBM              | SuperBlocking Top-15 | Target consistency         |
| EXP-04     | LightGBM              | SuperBlocking Top-15 | Source-specific thresholds |
| EXP-05     | Name-only ablation    | SuperBlocking Top-15 | Name features only         |
| EXP-06     | Address-only ablation | SuperBlocking Top-15 | Address features only      |
| EXP-07     | HistGradientBoosting  | SuperBlocking Top-15 | Alternative tree model     |
| EXP-08     | Deeper LightGBM       | SuperBlocking Top-15 | Model-depth experiment     |
| EXP-09     | Enhanced features     | SuperBlocking Top-20 | Expanded feature set       |
| EXP-10     | LGB + HGB ensemble    | SuperBlocking Top-20 | Ensemble experiment        |

The validation figures in the original experiment log should be interpreted as **experiment-specific results**, rather than as a description of the current production XGBoost implementation.

For example, the reported EXP-09 and EXP-10 results were:

| Experiment | Precision | Recall | Macro F0.5 |
| ---------- | --------: | -----: | ---------: |
| EXP-09     |    0.9924 | 0.9454 |   0.976105 |
| EXP-10     |    0.9932 | 0.9439 |   0.976653 |

These numbers belong to the documented validation experiments and should not be presented as the measured performance of the current XGBoost GPU production model unless that model is evaluated separately.

---

# 10. Error Analysis

Two major classes of matching errors were identified during the documented experiments.

### False positives

Typical false-positive cases include businesses with:

* Very similar or generic names
* Similar addresses
* Shared commercial buildings
* Different units within the same location

Address-number conflict features and name-conflict signals help reduce these incorrect merges.

### False negatives

Missed matches can occur when:

* The business name is heavily abbreviated
* A DBA/acronym replaces the original name
* The address is substantially corrupted
* Important address information is missing

These cases are difficult because both major evidence sources can be degraded simultaneously.

---

# 11. Implementation Structure

The main source tree is organized into separate modules for the individual stages of the pipeline.

```text
code/
└── business_entity_resolution/
    └── src/
        ├── normalization.py
        ├── blocking.py
        ├── features.py
        ├── model.py
        ├── thresholding.py
        ├── evaluation.py
        ├── data_loader.py
        ├── output.py
        ├── inference.py
        └── main.py
```

### Module responsibilities

| File               | Responsibility                                             |
| ------------------ | ---------------------------------------------------------- |
| `normalization.py` | Text cleaning, transliteration, name/address normalization |
| `blocking.py`      | Blocking-key construction and candidate retrieval          |
| `features.py`      | Pairwise feature extraction                                |
| `model.py`         | Model training, prediction, saving and loading             |
| `thresholding.py`  | Probability thresholding and target deduplication          |
| `evaluation.py`    | Evaluation and competition metrics                         |
| `data_loader.py`   | Dataset loading and preprocessing                          |
| `output.py`        | Submission and report generation                           |
| `inference.py`     | End-to-end test inference                                  |
| `main.py`          | Command-line entry point and validation                    |

---

# 12. Requirements

The current implementation requires the libraries used directly by the source code:

```text
numpy>=1.24,<3.0
xgboost>=2.0,<4.0
lightgbm>=4.0,<5.0
joblib>=1.3,<2.0
text-unidecode>=1.3,<2.0
```

XGBoost is required for the current production model path, while LightGBM remains available through the model wrapper.

---

# 13. Running the Pipeline

From the project root:

```bash
python code/business_entity_resolution/src/main.py \
    --test-dir student_resource/dataset/test \
    --output-dir output
```

Additional options can be supplied when required:

```bash
--model-path
--meta-path
--batch-size
--top-k
--validate
```

The pipeline produces the required matching output together with candidate and evaluation/report artefacts configured by the inference and output modules.

---

# 14. Key Design Principles

The solution is built around four main ideas:

### 1. Reduce the search space before classification

Blocking prevents the classifier from evaluating the complete Cartesian product of the datasets.

### 2. Preserve information through multiple representations

No single name or address similarity measure is sufficient for noisy entity resolution. Multiple normalized representations and similarity features are therefore combined.

### 3. Separate pairwise confidence from global consistency

The classifier evaluates individual candidate pairs, while the final deduplication stage applies constraints across competing matches.

### 4. Keep inference memory-efficient

Batch prediction, country partitioning, and disk-backed intermediate storage allow the pipeline to process large test datasets without keeping the complete candidate space in memory.

---

# 15. Conclusion

The final system follows a modular entity-resolution architecture:

```text
Normalization
      ↓
Country Partitioning
      ↓
Multi-Key Blocking
      ↓
Candidate Generation
      ↓
32-Dimensional Pair Features
      ↓
Gradient-Boosted Classification
      ↓
Source-Specific Thresholding
      ↓
Target Deduplication
      ↓
Submission Generation
```

This design separates **candidate retrieval**, **pairwise matching**, and **global consistency**, making the pipeline easier to experiment with and scale to large datasets.

The current production inference path uses **XGBoost with CUDA acceleration**, while the model abstraction also supports LightGBM. Validation experiments documented in the project are kept separate from production-model performance so that reported experimental scores are not incorrectly attributed to the current inference configuration.
