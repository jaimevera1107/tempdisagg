# ⚡️ tempdisagg

> **Temporal Disaggregation Models in Python — Modular · Robust · Ready for Production**

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/tests-100%25-success)
![PyPI - Status](https://img.shields.io/badge/pypi-ready-yellow)

---

`tempdisagg` is a Python library for **temporal disaggregation of time series data**.  
It supports all classical methods — **Chow-Lin**, **Litterman**, **Denton**, **Fernández**, **Uniform** — and offers a **modular, extensible, production-grade** architecture, inspired by the R package `tempdisagg`.

✨ The library combines:
- 📈 Regression-based models
- 📉 Differencing & smoothing techniques
- 🤖 Ensemble learning
- 🛠 Post-estimation adjustments
- 🧠 Full integration with the Python scientific stack

---

Many official statistics and business indicators are reported at low frequencies (e.g., annually or quarterly), but decision-making often demands **high-frequency estimates**.  
Temporal disaggregation bridges this gap by producing granular series that **preserve consistency with aggregate values**.

**`tempdisagg`** provides a flexible interface to solve this problem — using econometric, statistical and machine learning techniques in a unified Pythonic API.

---


## 📚 Methods Implemented

| Method(s)                                                                 | Description                                                   |
|---------------------------------------------------------------------------|---------------------------------------------------------------|
| `ols`                                                                     | Ordinary Least Squares (baseline)                             |
| `denton`, `denton-opt`                                                    | Denton interpolation with optional differencing               |
| `denton-cholette`                                                         | Modified smoother from Dagum & Cholette                       |
| `chow-lin`, `chow-lin-opt`, `chow-lin-ecotrim`, `chow-lin-quilis`        | Regression-based disaggregation with autoregressive adjustment |
| `litterman`, `litterman-opt`                                              | Litterman method with random walk / AR(1) prior               |
| `fernandez`                                                               | Second-order differencing (Litterman with ρ = 0)              |
| `fast`                                                                    | Fast approximation of `denton-cholette`                       |
| `uniform`                                                                 | Uniform distribution across subperiods                        |

---

## 🛠️ Installation

```bash
pip install tempdisagg
```

### 💡 Quick Example

```python
from tempdisagg import TempDisaggModel

# Create your DataFrame
df = pd.DataFrame({
    "Index": [2020]*12 + [2021]*12,
    "Grain": list(range(1, 13))*2,
    "y": [1200] + [np.nan]*11 + [1500] + [np.nan]*11,
    "X": np.linspace(100, 200, 24)
})

# Initialize and fit model
model = TempDisaggModel(method="chow-lin-opt", conversion="sum")
model.fit(df)

# Predict high-frequency series
y_hat = model.predict()

# Summary and plots
model.summary()
model.plot(df)
```
---

### 🤖 How does the Ensemble Prediction work?

The ensemble module allows combining multiple disaggregation methods into a single high-frequency estimate. It works by:

1. **Fitting multiple models** individually on the same input dataset using methods like `chow-lin`, `denton`, `fernandez`, etc.
2. **Computing the aggregated prediction error** (e.g., RMSE or MAE) of each model with respect to the low-frequency constraint.
3. **Optimizing weights** across models using non-negative least squares to minimize the error of the aggregated prediction (subject to weights summing to 1).
4. **Generating a final ensemble prediction**:  
   $$\hat{y}_{ensemble} = \sum_{i} w_i \cdot \hat{y}_i$$

   where $\hat{y}_i$ is the prediction of the $i$ -th model and $ w_i $ is its optimal weight.

Additional features:
- Bootstrap-based confidence intervals for the ensemble.
- Aggregated statistics such as average coefficients and combined R².
- Visual comparison of all component models via `.plot(df, show_individuals=True)`.

### 🤝 Ensemble Modeling
```python
model = temporal-dissagregationModel(conversion="average")
model.ensemble(df)

model.summary()
model.plot(df, use_adjusted=True)
```
---


### 🚫 How does the Negative Value Adjustment work?

Temporal disaggregation methods may produce negative high-frequency values when:
- The low-frequency total is small.
- The method uses strong differencing or extrapolation.
- Indicator series are noisy or weakly correlated.

To address this, `tempdisagg` applies a post-estimation adjustment that:

1. **Detects negative predictions** in `y_hat`.
2. **Groups values** by low-frequency periods using the conversion matrix $ C $.
3. **Redistributes residuals** within each group to ensure all values are non-negative and consistent:
   - Preserves the original sum (`C @ y_hat_adjusted = y_l`).
   - Applies proportional or uniform redistribution to correct negatives.
   - Ensures $\hat{y}_{adjusted} \geq 0$ without breaking constraints.

### ✅ Negatives Adjustment
```python
model = temporal-dissagregationModel(conversion="average")
model.predict(df)
model.adjust_output(df)
```


---


## 🗂️ Input Time Series Format

To use `TempDisModel`, your time series data must be organized in a **long-format DataFrame** with one row per high-frequency observation. The model requires the following columns:

| Column          | Description |
|-----------------|-------------|
| `Index`         | Identifier for the low-frequency group (e.g., year, quarter). This groups the target values. |
| `Grain`         | Identifier for the high-frequency breakdown within each `Index` (e.g., month, quarter number). |
| `y`             | The **low-frequency target variable** (repeated across the group). This is the variable to disaggregate. |
| `X`             | The **high-frequency indicator** variable (available at the granular level). Used to guide the disaggregation. |

---

#### 🔢 Example Structure

| Index | Grain | y       | X         |
|-------|-------|---------|-----------|
| 2000  | 1     | 1000.00 | 80.21     |
| 2000  | 2     | 1000.00 | 91.13     |
| 2000  | 3     | 1000.00 | 85.44     |
| 2000  | 4     | 1000.00 | 92.32     |
| 2001  | 1     | 1200.00 | 88.71     |
| 2001  | 2     | 1200.00 | 93.55     |
| ...   | ...   | ...     | ...       |

---



### ⚙️ API Overview

| Method                         | Description                                                  |
|-------------------------------|--------------------------------------------------------------|
| `.fit(df)`                    | Fit model to input DataFrame                                |
| `.predict()`                  | Return high-frequency `y_hat`                               |
| `.fit_predict(df)`            | Shortcut to `.fit().predict()`                              |
| `.summary(metric="mae")`      | Print summary with t-stats, AIC, BIC, R²                     |
| `.plot(df, use_adjusted=False)` | Plot predictions                                           |
| `.adjust_output(df)`          | Apply non-negative adjustment                              |
| `.ensemble(df, methods=...)`  | Fit ensemble and combine multiple models                    |
| `.validate_aggregation()`     | Check if `C @ y_hat ≈ y_l`                                  |
| `.get_params()` / `.set_params()` | Get/set model configuration                           |
| `.to_dict()`                  | Export results in serializable dictionary                  |



## 🧠 Modular Architecture

The codebase follows a clean architecture with decoupled components:

- `TempDisaggModel`: High-level API
- `ModelsHandler`: Implements individual disaggregation methods
- `RhoOptimizer`: Optimizes autocorrelation
- `DisaggInputPreparer`: Manages time series preparation
- `PostEstimation`: Adjusts predictions post-estimation
- `EnsemblePrediction`: Combines multiple models into one


### 🧪 Testing & Validation

The library includes:

- Unit tests for all modules
- Validation of input dimensions and types
- Aggregation consistency checks
- Support for NaNs and ragged time indices


## 🧩 **Related Projects**

**In R:**
- [`tempdisagg`](https://cran.r-project.org/package=tempdisagg) – Reference package for temporal disaggregation.

---

## 📚 **References and Acknowledgements**

This library draws inspiration from the R ecosystem and academic literature on temporal disaggregation.

Their research laid the foundation for many techniques implemented here.  
For a deeper review, we encourage exploring the reference section in the [`tempdisagg`](https://cran.r-project.org/package=tempdisagg) R package.

---

## 📃 **License**  
This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for more details.
