# AI-SIR: An Empirical Study of Cross-Cultural Diffusion under Generative AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen.svg)]()

> Complete implementation of an AI-SIR diffusion model for the cross-cultural propagation of Chinese civilization content on overseas TikTok, June 2024 – May 2025.

This repository contains the full empirical pipeline: data preprocessing, AI content detection with manual validation, propensity score matching, nonlinear least squares parameter estimation, multi-dimensional robustness checks, and all paper figures and tables.

---

## Table of Contents

- [Background](#background)
- [Model](#model)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Reproducing the Results](#reproducing-the-results)
- [Key Empirical Findings](#key-empirical-findings)
- [Data Description](#data-description)
- [Module Reference](#module-reference)
- [Citation](#citation)
- [License](#license)

---

## Background

The classical SIR (Susceptible – Infected – Recovered) compartmental model has been widely repurposed to describe information diffusion in social networks. With the rise of generative AI, however, classical SIR no longer captures the **algorithmic amplification** and **cross-cultural semantic attenuation** that now jointly shape how content circulates online.

This project implements the **AI-SIR** model, which extends classical SIR with two structural parameters:

- **α (AI amplification coefficient)** — quantifies the boost in effective contact rate produced by AI tools (automatic translation, recommendation algorithms, AI-generated summaries).
- **δ (cultural discount factor)** — captures the inevitable semantic attenuation when cultural content crosses linguistic and cognitive boundaries.

The empirical setting is 18,326 short videos themed on Chinese civilization (food, architecture, history, philosophy) posted on overseas TikTok between June 2024 and May 2025.

## Model

### Classical SIR

```
dS/dt = −β·SI/N
dI/dt =  β·SI/N − γ·I
dR/dt =  γ·I
```

### AI-SIR (this work)

```
dS/dt = −α·δ·β·SI/N
dI/dt =  α·δ·β·SI/N − γ·I
dR/dt =  γ·I
```

### Basic Reproduction Number

```
       α · δ · β
R₀ = ─────────────
           γ
```

When α=1 and δ=1, AI-SIR reduces to classical SIR.

### Estimation

The transmission rate β and recovery rate γ are estimated by nonlinear least squares (Trust-Region-Reflective):

```
min_{β,γ}  Σ_t [ I_obs(t) − I_model(t; β, γ, α, δ) ]²
```

The ODE system is integrated with the fourth-order Runge-Kutta method (truncation error O(h⁵), step size h=1 day).

## Repository Layout

```
AI_SIR_Diffusion_Model/
├── README.md                          ← this file
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── raw/
│   │   ├── AI_SIR_RawData.xlsx        ← 6 sheets, 18,326 videos
│   │   ├── Comments_Sentiment.csv     ← 280,617 XLM-RoBERTa-tagged comments
│   │   └── AI_Detection_Features.csv  ← GPTZero + Originality.ai features
│   ├── processed/                     ← intermediate caches (generated)
│   └── results/                       ← key result objects (generated)
│
├── src/                               ← algorithm implementation
│   ├── config.py                      ← global parameters and paths
│   ├── data_loader.py                 ← I/O + I(t) construction
│   ├── sir_model.py                   ← AI-SIR ODE + RK4 integrator
│   ├── parameter_estimation.py        ← NLS + DW + Ljung-Box
│   ├── preprocessing.py               ← Section 4.1
│   ├── ai_detection.py                ← Section 4.2.1 manual validation
│   ├── psm.py                         ← Section 3.5 propensity score matching
│   ├── content_analysis.py            ← Section 4.3.1 by content type
│   ├── regional_analysis.py           ← Section 4.3.1 regional δ
│   ├── sentiment_analysis.py          ← Section 4.3.2 polarity ANOVA
│   ├── sensitivity_analysis.py        ← Section 4.3.3 elasticity
│   ├── robustness_checks.py           ← Section 4.4 seven-dimensional checks
│   └── visualization.py               ← Figures 3–8
│
├── scripts/                           ← driver scripts (10 + master)
│   ├── 01_descriptive_stats.py
│   ├── 02_manual_validation.py
│   ├── 03_psm_matching.py
│   ├── 04_main_estimation.py
│   ├── 05_content_analysis.py
│   ├── 06_regional_analysis.py
│   ├── 07_sentiment_analysis.py
│   ├── 08_sensitivity.py
│   ├── 09_robustness.py
│   ├── 10_visualizations.py
│   ├── generate_report.py
│   └── run_all.py                     ← master orchestrator
│
└── outputs/
    ├── tables/                        ← all CSV tables (Tables 3–12)
    ├── figures/                       ← all PNG figures (Figures 3–8)
    └── reports/                       ← consolidated narrative report
```

## Quick Start

### Prerequisites

- Python 3.10 or newer
- 200 MB of free disk space (for outputs)

### Installation

```bash
git clone <repository-url>
cd AI_SIR_Diffusion_Model
pip install -r requirements.txt
```

Required packages (pinned in `requirements.txt`):

| Package | Minimum version | Purpose |
|---|---|---|
| numpy | 1.24 | Numerical arrays, RK4 |
| pandas | 2.0 | Tabular data, I/O |
| scipy | 1.10 | NLS, ODE backup solver, statistical tests |
| scikit-learn | 1.3 | Logistic PSM, KNN matching, metrics |
| matplotlib | 3.7 | Figures 3–8 |
| openpyxl | 3.1 | XLSX I/O |

### Run Everything

```bash
python scripts/run_all.py
```

The full pipeline completes in roughly **60–70 seconds** on a single CPU. All tables and figures are written to `outputs/`.

### Single-step Usage

```python
from src.data_loader import load_videos, build_It_from_videos
from src.parameter_estimation import fit_ai_sir

videos = load_videos()
I_obs = build_It_from_videos(videos)

result = fit_ai_sir(I_obs, alpha=1.465, delta=0.618)
print(f"β  = {result['beta']:.3f}")
print(f"γ  = {result['gamma']:.3f}")
print(f"R₀ = {result['R0']:.3f}")
print(f"R² = {result['R2']:.3f}")
```

## Reproducing the Results

Run any section in isolation:

```bash
python scripts/01_descriptive_stats.py     # Section 4.1
python scripts/02_manual_validation.py     # Section 4.2.1 manual validation
python scripts/03_psm_matching.py          # Section 3.5 + 4.2.1 PSM
python scripts/04_main_estimation.py       # Section 4.2.2 main NLS
python scripts/05_content_analysis.py      # Section 4.3.1 Table 6
python scripts/06_regional_analysis.py     # Section 4.3.1 Table 7
python scripts/07_sentiment_analysis.py    # Section 4.3.2 Table 8
python scripts/08_sensitivity.py           # Section 4.3.3 sensitivity
python scripts/09_robustness.py            # Section 4.4 Tables 9–12
python scripts/10_visualizations.py        # Figures 3–8
python scripts/generate_report.py          # Consolidated text report
```

The numerical pipeline is fully deterministic — a fixed global seed is used for any stochastic step.

## Key Empirical Findings

### Main Parameter Estimates (full sample)

| Parameter | Value | 95% CI | Interpretation |
|---|---|---|---|
| β | 0.129 | [0.117, 0.141] | Baseline transmission rate per day |
| γ | 0.057 | [0.049, 0.065] | Recovery rate per day (≈17.5-day active window) |
| α | 1.465 | — | AI amplification (naive ratio) |
| δ | 0.618 | [0.583, 0.653] | Cultural discount factor (full sample mean) |
| **R₀** | **2.053** | [1.826, 2.297] | **Basic reproduction number** |
| R² | 0.930 | — | Coefficient of determination |
| RMSE | 1,341 | — | Root mean squared error |

After **propensity score matching** on seven creator-level covariates, the net AI amplification is α=1.32 (95% CI [1.21, 1.43]) and R₀=1.78, both still comfortably above the critical threshold for sustained diffusion.

### Heterogeneity across Content Types

| Content type | α | δ | β | γ | **R₀** |
|---|---|---|---|---|---|
| Food | 1.512 | 0.738 | 0.141 | 0.056 | **2.81** |
| Architecture & Scenery | 1.483 | 0.672 | 0.133 | 0.058 | **2.29** |
| Historical & Humanistic | 1.427 | 0.521 | 0.124 | 0.061 | **1.51** |
| Philosophy & Cultural Symbols | 1.384 | 0.407 | 0.118 | 0.064 | **1.04** |

Philosophy sits at the diffusion critical threshold — any minor parameter deterioration would break the propagation chain.

### Regional Cultural Discount

| Sub-region | Share of non-English audience | δ |
|---|---|---|
| East Asia (JP, KR) | 26.5% | **0.775** |
| Southeast Asia | 32.4% | **0.686** |
| Continental Europe | 19.1% | **0.510** |
| Latin America | 13.2% | **0.452** |
| Middle East / Others | 8.8% | **0.380** |
| **Audience-weighted mean** | 100% | **0.618** |

The gradient cannot be explained by linguistic distance alone — it reflects deeper **cultural-cognitive proximity** to the source culture.

### Sentiment-driven Recovery Rate

Negative-sentiment-dominated content elevates γ by **+52.9%** relative to positive-sentiment content. The effect is largest for cognitively demanding categories (philosophy: +66.1%).

### Sensitivity Analysis

| Parameter | Elasticity w.r.t. R₀ |
|---|---|
| α | +1.000 |
| δ | +1.000 |
| β | +1.000 |
| γ | **−1.042** |

A 20% reduction in γ increases R₀ by 25.0% and peak diffusion magnitude substantially, identifying the **active dissemination window** as the most effective lever for amplifying overall reach.

### Robustness

Seven robustness dimensions all confirm R₀ > 1:

- 70/30 temporal cross-validation: train R²=0.944, val R²=0.907
- AI detection threshold (55%–85%): R₀ varies between 1.63–1.90
- Recovery-state definition (3–14 inactive days): R₀ varies between 1.19–2.25
- September–November policy-window exclusion: R₀ falls only from 1.78 to 1.61
- Time-decay augmentation: λ=0.0031, half-life=224 days
- Degree-corrected network SIR benchmark: amplification 1.36, R₀=1.84

## Data Description

### `AI_SIR_RawData.xlsx` (4.4 MB, 6 sheets)

| Sheet | Rows | Content |
|---|---|---|
| 1_字段说明 | 27 | Field dictionary |
| 2_视频原始数据 | 18,326 | Video-level metadata, engagement, audience composition |
| 3_创作者元数据 | 3,850 | Creator-level covariates (for PSM) |
| 4_标签词清单 | 97 | Search hashtag taxonomy |
| 5_人工标注样本300 | 300 | Stratified manual validation subsample |
| 6_日时间序列 | 365 | Daily aggregated time series |

### `Comments_Sentiment.csv` (36 MB)

280,617 comments tagged by XLM-RoBERTa cross-lingual sentiment classifier (positive / neutral / negative).

### `AI_Detection_Features.csv` (3.5 MB)

Per-video features from GPTZero (primary) and Originality.ai (auxiliary): probability scores, perplexity, burstiness, n-gram repetition, language confidence.

## Module Reference

### `sir_model.py`

Implements the AI-SIR right-hand side, an explicit RK4 integrator, and the scipy `solve_ivp` backup. Includes `compute_R0()` and `find_peak()` utilities.

### `parameter_estimation.py`

Wraps `scipy.optimize.least_squares` with bounds, returns parameter estimates, standard errors via the Jacobian, delta-method CI for R₀, and condition number. Also exposes Durbin-Watson and Ljung-Box statistics.

### `psm.py`

Builds the seven-dimensional covariate matrix, fits the propensity score with `LogisticRegression`, performs 1:1 nearest-neighbor matching with a 0.05-SD caliper, checks standardized mean differences, and re-estimates the AI amplification on the matched sample.

### `sensitivity_analysis.py`

Sweeps each parameter through ±20% in five steps, records R₀, peak timing, and peak magnitude, and computes elasticity coefficients.

### `robustness_checks.py`

Performs the full seven-dimensional robustness battery: cross-validation, AI threshold sensitivity, recovery-state threshold sensitivity, policy-window exclusion, time-decay augmentation, and the network SIR external benchmark.

### `visualization.py`

Reproduces Figures 3–8 in publication quality (150 DPI, vector-friendly typography).

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{aisir2025,
  title   = {Generative AI and the Cross-Cultural Diffusion of Chinese
             Civilization: An AI-SIR Empirical Study},
  journal = {Applied Sciences},
  year    = {2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- TikTok Research API for the underlying engagement data.
- GPTZero and Originality.ai for AI-content detection scores.
- The XLM-RoBERTa team for the multilingual sentiment classifier.

---

*Questions or issues? Please open an issue on GitHub.*
