# ⚡️ **tempdisagg**

>### **Temporal Disaggregation Models in Python**

*High-Frequency Estimation from Low-Frequency Data — Modular · Robust · Ready for Production*

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/tests-100%25-success)
![PyPI](https://img.shields.io/pypi/v/tempdisagg)

---

`tempdisagg` is a production-ready Python library for **temporal disaggregation of time series** — transforming low-frequency data into high-frequency estimates while preserving consistency.

> It supports all major classical methods — **Chow-Lin**, **Litterman**, **Denton**, **Fernández**, **Uniform** — and provides a **clean modular architecture** inspired by R's `tempdisagg`, with modern additions:

- 📈 Regression + autoregressive adjustment  
- 📉 Differencing & smoothing interpolators  
- 🤖 Ensemble prediction engine  
- 🧠 Intelligent padding & interpolation  
- 🔧 Post-estimation fix for negative values  
- 🔄 Optional retropolarization via regression    

---

## 🔍 Why Temporal Disaggregation?

Official indicators often come in low frequency (e.g. yearly GDP), while economic analysis or forecasting needs monthly or quarterly granularity. `tempdisagg` fills this gap using consistent econometric techniques to create **granular estimates** that **respect original aggregates**.

---

## 📚 Methods Implemented

| Method(s)                                                               | Description                                                   |
|-------------------------------------------------------------------------|---------------------------------------------------------------|
| `ols`                                                                   | Ordinary Least Squares (baseline)                             |
| `denton`                                                                | Denton interpolation with differencing                        |
| `denton-cholette`                                                       | Cholette smoother variant from Dagum & Cholette               |
| `chow-lin`, `chow-lin-opt`, `chow-lin-ecotrim`, `chow-lin-quilis`       | Regression + AR(1) residual modeling                          |
| `litterman`, `litterman-opt`                                            | Random walk / AR(1) prior models                              |
| `fernandez`                                                             | Fixed-ρ Litterman (ρ = 0)                                     |
| `fast`                                                                  | Fast approximation of Denton-Cholette                         |
| `uniform`                                                               | Even distribution across subperiods                           |

---

## 💾 Installation

```bash
pip install tempdisagg
```

---

## 🚀 Quick Example

```python
from tempdisagg import TempDisaggModel
import pandas as pd
import numpy as np

# Sample input data (monthly disaggregation of yearly total)
df = pd.DataFrame({
    "Index": [2020]*12 + [2021]*12,
    "Grain": list(range(1, 13)) * 2,
    "y": [1200] + [np.nan]*11 + [1500] + [np.nan]*11,
    "X": np.linspace(100, 200, 24)
})

# Fit model
model = TempDisaggModel(method="chow-lin-opt", conversion="sum")
model.fit(df)

# Predict high-frequency series
y_hat = model.predict()

# Adjust negatives (if any; OPTIONAL)
y_adj = model.adjust_output()

# Show results
model.summary()
model.plot()
````

---

## ⚡ Example with Real Data

```python
import statsmodels.api as sm
from tempdisagg import TempDisaggModel

# Load macroeconomic dataset (quarterly)
macro = sm.datasets.macrodata.load_pandas().data
macro["Index"] = macro["year"].astype(int)
macro["Grain"] = macro["quarter"].astype(int)
macro["X"] = macro["realcons"]

# Aggregate GDP to annual level
gdp_annual = macro.groupby("Index")["realgdp"].mean().reset_index()
gdp_annual.columns = ["Index", "y"]

# Merge back into full frame
df = macro.merge(gdp_annual, on="Index", how="left")[["Index", "Grain", "y", "X"]]

# Fit model and predict
model = TempDisaggModel(method="chow-lin-opt", conversion="average")
model.fit(df)

# Get high-frequency estimates
y_hat = model.predict(full=False)

# Optional: post-estimation adjustment
y_adj = model.adjust_output(full=False)

# Summary and plot
model.summary()
model.plot()
```

---

## 🤖 Ensemble Prediction

Run all models and let the library **find the optimal weighted combination**.

```python
model = TempDisaggModel(method="ensemble", conversion="sum")
model.fit(df)

model.summary()
model.plot()
```

Behind the scenes:
- Each method is fitted separately.
- Error metrics (e.g. MAE) are computed.
- Weights are optimized to minimize global error.
- Final prediction is a weighted average across models.

---

## 🚫 Negative Value Adjustment

When disaggregation outputs negatives (due to smoothing or regression noise), `tempdisagg` can correct them **without violating consistency**.

```python
model.fit(df)
y_hat = model.adjust_output()
```

Internally:
- Detects negatives in each group.
- Redistributes values proportionally.
- Ensures aggregate values match original data.

---

## 🧠 Retropolarizer: Smart Interpolation

For missing values in the target (`y`), you can activate the **Retropolarizer**: a module that imputes via regression, proportions, or exponential smoothing.

```python
from tempdisagg import Retropolarizer
retro = Retropolarizer(df = data, new_col = "new", old_col = "old")
df["y_imputed"] = retro.retropolarize(method='proportion')
```

Or use it inside any model:

```python
model = TempDisaggModel(
    method="chow-lin",
    use_retropolarizer=True,
    retro_method="linear_regression"
)
model.fit(df)
```

Available methods:

- 'proportion'
- 'linear_regression'
- 'polynomial_regression' 
- 'exponential_smoothing'
- 'mlp_regression'

> **Note:** The Retropolarizer is only used to impute missing values in the `y` column.  It is **not** intended for interpolating the `X` (indicator) variable.


---

## 📘 Input Format

Your data must be in long format:

| Column   | Meaning                                        |
|----------|------------------------------------------------|
| `Index`  | Low-frequency group ID (e.g., year)            |
| `Grain`  | High-frequency unit (e.g., month number)       |
| `y`      | Target variable (repeated within group)        |
| `X`      | Indicator variable at high frequency           |

```text
Index | Grain | y     | X
------|-------|-------|-----
2020  | 1     | 1000  | 10.1
2020  | 2     | 1000  | 11.3
2020  | 3     | 1000  | 12.5
...   | ...   | ...   | ...
```

---

## 🧩 Modular Design

| Component              | Role                                       |
|------------------------|--------------------------------------------|
| `TempDisaggModel`      | High-level interface                       |
| `DisaggInputPreparer`  | Input validation + padding + interpolation |
| `ModelsHandler`        | Implements disaggregation methods          |
| `RhoOptimizer`         | Optimizes AR(1) parameter                  |
| `PostEstimation`       | Adjusts negative values                    |
| `EnsemblePrediction`   | Combines multiple models                   |
| `Retropolarizer`       | Regression-based imputer for `y`           |

---

## 🧪 Testing & Reliability

- ✅ Full test coverage  
- ✅ Input validation & fallbacks  
- ✅ Padding & missing data supported  
- ✅ Consistency validation `C @ y_hat ≈ y_l`  

---

## 🔍 API Overview

| Method                         | Description                                         |
|--------------------------------|-----------------------------------------------------|
| `.fit(df)`                     | Fit the model                                       |
| `.predict(full=True)`          | Predict disaggregated values                        |
| `.adjust_output(full=True)`    | Fix negative predictions                            |
| `.summary(metric="mae")`       | Print coefficients, rho, and errors                 |
| `.plot(use_adjusted=False)`    | Visualize predictions                               |
| `.get_params()` / `.set_params()` | Get/set model config                            |
| `.to_dict()`                   | Export results                                      |

---

## 📦 Dependencies

`tempdisagg` relies on the following Python libraries:

- `pandas` – data manipulation  
- `numpy` – numerical operations  
- `matplotlib` – plotting  
- `scipy` and `statsmodels` – regression and optimization  
- `scikit-learn` – used in `Retropolarizer` (e.g., MLP imputation)

> These packages are automatically installed with `pip install tempdisagg`.

---

## 📄 Cite this work


This project is accompanied by a scientific publication available on **arXiv**:

> Jaime A. Jaramillo-Vera. (2025). *tempdisagg: A Python Library for Temporal Disaggregation of Time Series*. arXiv:2503.22054 [econ.EM].  
> [🔗 View on arXiv](https://arxiv.org/abs/2503.22054)

If you use this library in your research, please cite the paper to support the development and visibility of the project.

```bibtex
@misc{verajaramillo2025tempdisagg,
  title        = {tempdisagg: A Python Library for Temporal Disaggregation of Time Series},
  author       = {Jaime A. Jaramillo-Vera},
  year         = {2025},
  eprint       = {2503.22054},
  archivePrefix = {arXiv},
  primaryClass = {econ.EM},
  url          = {https://arxiv.org/abs/2503.22054}
}
```
---

## 📚 References

- Dagum & Cholette (2006), *Benchmarking, Temporal Distribution, and Reconciliation Methods*  
- Denton (1971), *Adjustment of Monthly or Quarterly Series*  
- Chow & Lin (1971), *Best Linear Unbiased Estimation of Missing Observations*  
- Fernández (1981), *Methodological Note on a Monthly Indicator*  
- Litterman (1983), *A Random Walk, Markov Model for Forecasting*
- tempdisagg (R package)

---

## 📃 License

MIT License — See [LICENSE](./LICENSE) for details.


Developed and maintained by Jaime Vera-Jaramillo — Contributions are welcome ❤️.
