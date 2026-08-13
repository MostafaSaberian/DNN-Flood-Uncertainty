# Uncertainty Quantification of Deep Neural Networks for Flood Prediction

This repository contains the **data and source codes** associated with the manuscript:

**"Insights into the Uncertainties of Deep Neural Networks for Flood Prediction"**

**Authors:** Mostafa Saberian, Vidya Samadi, Thorsten Wagener, and Ioana Popescu

The repository provides the datasets, deep neural network implementations, uncertainty quantification methods, sensitivity analyses, and evaluation scripts used to reproduce the experiments presented in the manuscript.

---

## Overview

Deep neural networks (DNNs) have demonstrated strong predictive performance for hydrologic forecasting; however, reliable characterization of predictive uncertainty remains important for their application to flood prediction.

This study evaluates uncertainty associated with three DNN architectures:

* **Long Short-Term Memory (LSTM)**
* **Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)**
* **Neural Basis Expansion Analysis for Interpretable Time Series Forecasting (N-BEATS)**

Two approaches were used to characterize **model uncertainty**:

* Monte Carlo Dropout (MC Dropout)
* Markov Chain Monte Carlo (MCMC)

Two approaches were used to characterize **data uncertainty**:

* Quantile loss
* Heteroscedastic loss

The combinations of these methods were evaluated across three hydrologically distinct watersheds in the southeastern United States.

---

## Case Studies

The experiments were conducted for three watersheds representing different hydrological response characteristics.

### 1. Killian Creek, North Carolina

* **Watershed:** Upper Dutchmans Creek
* **USGS station:** 0214269560
* Represents a small, rapidly responding headwater watershed.
* Characterized by relatively short response times and rapid flood hydrographs.

### 2. Dog River, Georgia

* **Watershed:** Lower Dog River
* **USGS station:** 02337410
* Represents an upper watershed tributary with intermediate hydrological response characteristics.

### 3. Little Pee Dee River, South Carolina

* **USGS station:** 02135000
* Represents a larger coastal plain watershed.
* Characterized by slower hydrological response, substantial storage effects, and prolonged flood durations.

These contrasting watershed characteristics were used to investigate how uncertainty estimation methods behave under different hydrological regimes.

---

## Repository Structure

```text
DNN-Flood-Uncertainty/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│
├── source_codes/
│   ├── DogRiver/
│   ├── KillianCreek/
│   └── LittlePeeDee/

```

---

## Data

The `data/` directory contains the datasets used for model development and evaluation for the three case studies.

The hydrological discharge observations used in this study were obtained from the **U.S. Geological Survey (USGS)**.

The datasets were preprocessed before model training. Missing discharge observations represented less than 1% of the time series and were handled using linear interpolation.

Independent storm events were identified using the **Minimum Inter-Event Time (MIT)** approach.

Users should refer to the manuscript for detailed information regarding data sources, preprocessing procedures, study periods, and flood-event selection.

---

## Source Codes

The `source_codes/` directory contains the scripts used to train, evaluate, and analyze the uncertainty of the DNN models.

```text
source_codes/
├── DogRiver/
├── KillianCreek/
└── LittlePeeDee/
```
---

## Uncertainty Quantification Configurations

Six principal uncertainty configurations were evaluated for each DNN architecture:

1. **MC Dropout**
2. **MC Dropout + Quantile Loss**
3. **MC Dropout + Heteroscedastic Loss**
4. **MCMC**
5. **MCMC + Quantile Loss**
6. **MCMC + Heteroscedastic Loss**

MC dropout and MCMC were used to characterize uncertainty associated with model parameters, whereas quantile and heteroscedastic loss formulations were used to represent data uncertainty.

---

## Sensitivity Analysis

Sensitivity analyses were performed to investigate the effects of the governing uncertainty parameters.

For **MC Dropout**, the following dropout rates were evaluated:

```text
0.10
0.20
0.30
```

For **MCMC**, the following prior standard deviations were evaluated:

```text
0.10
0.20
0.50
0.80
```

Based on the experiments presented in the manuscript, a dropout rate of **0.20** and an MCMC prior standard deviation of **0.20** provided a favorable balance between predictive interval coverage and sharpness across the case studies.

---

## Model Configuration

The primary optimized model settings used in the experiments included:

| Parameter     |  LSTM | N-HiTS | N-BEATS |
| ------------- | ----: | -----: | ------: |
| Epochs        |    50 |     50 |      50 |
| Learning rate | 0.001 |  0.001 |   0.001 |
| Batch size    |    32 |     32 |      32 |
| Input window  |  24 h |   24 h |    24 h |
| Hidden units  |   128 |    128 |     128 |
| Activation    |  tanh |   ReLU |    ReLU |

For N-HiTS and N-BEATS, the optimized architectures contained three stacks with two blocks per stack.

Additional model-specific parameters can be found directly in the corresponding source-code files.

---

## Evaluation Metrics

Deterministic predictive performance was evaluated using:

* **Nash-Sutcliffe Efficiency (NSE)**
* **Kling-Gupta Efficiency (KGE)**

Predictive uncertainty was evaluated using:

* **P-factor**
* **R-factor**
* **Total Uncertainty Index (TUI)**

The P-factor measures the proportion of observed discharge values contained within the predictive uncertainty interval.

The R-factor represents the relative width of the predictive uncertainty interval compared with the variability of observed discharge.

TUI combines coverage and interval width to evaluate the trade-off between reliability and sharpness.

---

## Citation

If you use the data or codes provided in this repository, please cite the associated manuscript:

```bibtex
@article{saberian2026uncertainty,
  title   = {Insights into the Uncertainties of Deep Neural Networks for Flood Prediction},
  author  = {Saberian, Mostafa and Samadi, Vidya and Wagener, Thorsten and Popescu, Ioana},
  year    = {2026}
}
```

The complete citation, DOI, journal name, volume, and page information will be added following publication.

---

## Authors

**Mostafa Saberian**
The Glenn Department of Civil Engineering
Clemson University
Clemson, South Carolina, USA

**Vidya Samadi**
Clemson University

**Thorsten Wagener**
University of Potsdam

**Ioana Popescu**
IHE Delft Institute for Water Education / TU Delft

---

## Contact

For questions regarding the repository or the associated study, please contact:

**Mostafa Saberian**
Clemson University
Glenn Department of Civil Engineering
mostafs@clemson.edu

