# Actuarial Policy Persistency Modelling

An end-to-end actuarial analytics project investigating life insurance policy persistency and lapse risk using R.

The project develops an interpretable logistic regression framework to assess relative lapse risk, evaluates model performance under a highly imbalanced outcome, and demonstrates how predicted risk can support policyholder risk segmentation and retention prioritisation.

> **Important:** The lapse outcome in this project is **synthetic**, generated from predefined modelling assumptions. The results demonstrate an actuarial modelling workflow and the ability of the model to recover relationships embedded in the simulated portfolio. They should not be interpreted as empirical evidence of real-world lapse behaviour.

---

## 📌 Project Overview

Policy persistency is an important consideration in life insurance because policyholder behaviour can affect expected future premium income, cash flows, valuation assumptions, and retention strategy.

This project develops a complete workflow for identifying policies with relatively higher predicted lapse risk.

**Central question:**

> Can policyholder characteristics be used to rank policies by relative lapse risk in a way that could support actuarial analysis and targeted retention activity?

The project focuses on **prediction and risk ranking**, not causal inference. An association identified by the model does not imply that changing a particular characteristic would cause lapse behaviour to change.

---

## 📊 Dataset

The dataset contains 10,000 synthetic life insurance policyholder records.

| Portfolio characteristic | Value |
|---|---|
| Active policies | 9,646 |
| Lapsed policies | 354 |
| Overall lapse rate | 3.54% |

The dataset contains demographic, behavioural, health, product, and financial characteristics.

Data preparation includes:

- Missing-value assessment
- Duplicate checks
- Variable-type and category assessment
- Treatment of identifier variables
- Exclusion of unsuitable free-text / high-cardinality variables
- Transformation of policy start information into policy duration
- Construction of a premium-to-income ratio

The lapse target is generated synthetically using predefined assumptions. This limitation is documented throughout the project.

---

## 🔬 Methodology

The analysis follows an end-to-end actuarial modelling workflow.

### 1. Data Quality & Preparation
The dataset is assessed for completeness, duplication, variable structure, and suitability for modelling.

### 2. Actuarial Assumptions & Feature Engineering
Candidate representations are evaluated using both statistical evidence and actuarial interpretability. Key considerations include:

- Age representation
- Monthly premium versus premium-to-income ratio
- Policy duration
- Categorical policyholder characteristics

### 3. Exploratory Data Analysis
Exploratory analysis examines lapse behaviour across demographic, policy, behavioural, and financial characteristics, to identify patterns that inform subsequent modelling and actuarial interpretation.

### 4. Policy Persistency Analysis
Policy duration and persistency are examined to investigate how lapse behaviour varies over the policy lifecycle. An illustrative Kaplan–Meier analysis is also included.

> **Note:** genuine, independently observed lapse event times are unavailable in the synthetic dataset. The survival analysis is therefore illustrative and is not treated as evidence of real-world survival behaviour.

### 5. Predictive Modelling
Four logistic regression specifications are developed:

1. Null / baseline model
2. Raw-variable model
3. Business-engineered model
4. Stepwise AIC-simplified model

The final model prioritises a balance between predictive performance, parsimony, and interpretability.

**Final model specification:**

```r
Lapsed ~ age_group + smoking_status + policy_type + premium_income_ratio
```

### 6. Model Validation
The final model is evaluated using:

- ROC / AUC discrimination
- 10-fold cross-validation
- Calibration
- Multicollinearity diagnostics
- Threshold-based classification metrics
- Risk ranking and lift

Because lapse is a rare outcome, the conventional 0.50 probability threshold is **not** treated as an appropriate basis for judging model usefulness.

### 7. Risk Segmentation
Predicted lapse probabilities are used to rank policies into ten risk deciles. Observed lapse rates and lift are then examined across the risk groups to assess whether the model concentrates a greater proportion of observed lapses within higher-risk segments.

---

## 📈 Exploratory Analysis

The project generates **13 figures** from the R workflow, saved to `figures/`.

**Portfolio & Exploratory Analysis**

| File | Figure |
|---|---|
| `00_target_class_distribution.png` | Target class distribution (Active vs. Lapsed) |
| `01_lapse_rate_by_characteristics.png` | Lapse rate by policy type, smoking status, health status, marital status |
| `02_lapse_rate_by_age_group.png` | Lapse rate by age group |
| `03_lapse_rate_by_duration.png` | Lapse rate by policy duration band |
| `04_numeric_distributions.png` | Age and monthly premium distributions by lapse status |
| `05_boxplots_affordability_tenure.png` | Boxplots — premium-to-income ratio, annual income, policy duration |
| `06_correlation_matrix.png` | Correlation matrix of numeric variables |

**Persistency Analysis**

| File | Figure |
|---|---|
| `07_persistency_curve_by_policy_type.png` | Illustrative Kaplan–Meier persistency curve by policy type |

**Validation & Risk Segmentation**

| File | Figure |
|---|---|
| `08_roc_curve.png` | ROC curve — final logistic regression model |
| `09_lift_chart.png` | Lift chart by predicted risk decile |
| `10_deviance_residuals.png` | Deviance residuals vs. fitted values |
| `11_calibration_plot.png` | Calibration plot (predicted vs. observed, by decile) |

**Predictive Modelling — Final Model Interpretation**

| File | Figure |
|---|---|
| `12_odds_ratio_forest_plot.png` | Odds ratio forest plot — final model |

---

## 🔎 Key Findings

### Model Development

The modelling progression demonstrates that a more complex feature set does not automatically produce materially better predictive performance.

| Model | AIC | BIC | Parameters | Test AUC |
|---|---|---|---|---|
| Null / Baseline | 2145.9 | 2152.8 | 1 | 0.500 |
| Raw Variables | 2115.4 | 2252.5 | 20 | 0.643 |
| Business Engineered | 2119.2 | 2270.0 | 22 | 0.635 |
| **Stepwise — Final** | **2103.8** | **2179.1** | **11** | **0.639** |

The final model achieves the lowest AIC while using substantially fewer parameters than the larger models. Although the raw-variable model has a marginally higher test AUC, the difference is small — the final model was retained as a more parsimonious and interpretable specification.

### Model Performance

| Metric | Result |
|---|---|
| Test AUC | 0.639 |
| 10-fold Cross-Validation ROC | ≈ 0.620 |
| Hosmer–Lemeshow p-value | ≈ 0.68 |
| Maximum VIF | ≈ 1.3 |

The AUC indicates modest discrimination. The model should therefore be interpreted primarily as a **risk-ranking framework**, rather than as a highly accurate individual lapse prediction system.

### Class Imbalance

Only approximately 3.5% of policies lapse.

At the default 0.50 threshold:

| Metric | Result |
|---|---|
| Accuracy | 96.5% |
| Sensitivity | 0.0% |
| Specificity | 100.0% |

The high accuracy is misleading — the model predicts almost no policies as lapsing.

Using an illustrative threshold of approximately 0.04:

| Metric | Result |
|---|---|
| Accuracy | 60.5% |
| Sensitivity | 62.3% |
| Specificity | 60.4% |

This demonstrates why threshold selection and appropriate evaluation metrics matter for rare-event actuarial modelling.

---

## 🎯 Risk Segmentation

The model demonstrates useful separation in relative risk.

The highest-risk decile has:

- **6.35%** observed lapse rate
- **3.53%** portfolio-average lapse rate
- **1.80×** lift
- **6.98%** mean predicted lapse probability

The top 20% predicted-risk segment has an observed lapse rate of approximately **5.8%**, compared with approximately **3.5%** for the overall portfolio.

The model should be viewed as a **risk-prioritisation tool**, not a mechanism for identifying with certainty which individual policyholders will lapse.

---

## 🧮 Final Model Interpretation

Reference categories: Age 18–29 · Non-smoker · Term Life

| Variable | Odds Ratio | p-value |
|---|---|---|
| Age 40–49 | 0.622 | 0.013 |
| Age 50–59 | 0.410 | <0.001 |
| Age 60+ | 0.281 | <0.001 |
| Current smoker | 1.974 | 0.002 |
| Universal Life | 0.659 | 0.028 |
| Whole Life | 0.412 | <0.001 |
| Premium-to-income ratio | 1215.61 | 0.017 |

Relative to the reference categories, age groups 40–49, 50–59 and 60+, Universal Life, and Whole Life are associated with lower estimated odds of lapse, while current smoking status and premium-to-income ratio are associated with higher estimated odds.

These are **model associations within the synthetic dataset**, not causal effects or empirical insurer experience.

> **Note on premium-to-income ratio:** the large odds ratio reflects the scale of the variable (a continuous ratio, not a percentage), and its wide confidence interval indicates substantial estimation uncertainty. The coefficient should not be interpreted as a precise magnitude of effect — see `output/final_model_coefficients.csv` and the R script for the per-percentage-point interpretation.

---

## 💼 Actuarial Application

The modelling framework provides a conceptual workflow for supporting retention decision-making:

```
Policyholder Data
       ↓
Predicted Lapse Probability
       ↓
Risk Ranking / Deciles
       ↓
Identify Higher-Risk Segments
       ↓
Prioritise Retention Resources
       ↓
Test Retention Interventions
       ↓
Monitor Outcomes
```

In a real insurance setting, predicted lapse risk could potentially be combined with:

- Policy value
- Customer characteristics
- Intervention cost
- Expected benefit
- Business constraints

to determine where retention resources could potentially be prioritised.

However, predictive modelling alone does not establish that a particular retention intervention will reduce lapse. Intervention testing would be required.

---

## 📘 IFRS 17 Relevance

Persistency assumptions can influence projected future insurance cash flows and may therefore be relevant to actuarial valuation processes. This project considers that relevance **conceptually**.

It does **not** calculate:

- IFRS 17 insurance liabilities
- Contractual Service Margin (CSM)
- Loss components

The IFRS 17 discussion is conceptual rather than a financial reporting calculation.

---

## ⚠️ Limitations

**Synthetic outcome.** The lapse target is simulated from predefined assumptions rather than derived from observed insurer experience. The analysis demonstrates whether the modelling workflow can recover relationships embedded in the simulated data, rather than discovering empirical lapse drivers.

**Modest predictive performance.** A test AUC of approximately 0.639 represents modest discrimination. The model is more appropriately interpreted as a risk-ranking framework than a highly accurate individual prediction system.

**Class imbalance.** Only approximately 3.5% of policies lapse. Consequently, conventional threshold-based accuracy can give a misleading impression of model performance.

**Survival-analysis limitation.** The Kaplan–Meier analysis is illustrative because genuine, independently observed lapse event times are unavailable.

**Prediction is not causality.** Associations identified by the model should not be interpreted as causal relationships. A higher estimated lapse risk for a particular characteristic does not establish that changing that characteristic would reduce lapse.

**Generalisability & deployment.** A real insurer would require observed experience data, temporal validation, external validation, appropriate model governance, model-drift monitoring, fairness assessment, and business-impact assessment before production deployment.

---

## 🚀 Future Development

Potential extensions include:

- Genuine event-time survival modelling
- Dynamic policyholder covariates
- Nonlinear model benchmarks
- Interaction modelling
- Uplift / causal modelling
- External validation
- More robust temporal validation
- Cost-sensitive retention optimisation

These extensions could move the analysis towards a more realistic, production-oriented persistency framework.

---

## 📁 Repository Structure

```
actuarial-policy-persistency/
│
├── README.md
│
├── data/
│   └── life_insurance_retention_dataset.csv
│
├── persistency_modelling.R
│
├── figures/
│   ├── 00_target_class_distribution.png
│   ├── 01_lapse_rate_by_characteristics.png
│   ├── 02_lapse_rate_by_age_group.png
│   ├── 03_lapse_rate_by_duration.png
│   ├── 04_numeric_distributions.png
│   ├── 05_boxplots_affordability_tenure.png
│   ├── 06_correlation_matrix.png
│   ├── 07_persistency_curve_by_policy_type.png
│   ├── 08_roc_curve.png
│   ├── 09_lift_chart.png
│   ├── 10_deviance_residuals.png
│   ├── 11_calibration_plot.png
│   └── 12_odds_ratio_forest_plot.png
│
├── output/
│   ├── model_comparison.csv
│   ├── final_model_coefficients.csv
│   ├── validation_results.csv
│   └── risk_segmentation.csv
│
└── report/
    └── Actuarial_Policy_Persistency_Modelling.pdf
```

---

## 📂 Repository Contents

**`data/`**
Contains the synthetic life insurance dataset used for the analysis.

**`persistency_modelling.R`**
The complete R modelling workflow, from data preparation through exploratory analysis, modelling, validation, and output generation.

**`figures/`**
Contains the 13 analytical figures generated by the R workflow.

**`output/`**
Contains the key numerical outputs:
- `model_comparison.csv`
- `final_model_coefficients.csv`
- `validation_results.csv`
- `risk_segmentation.csv`

**`report/`**
Contains the full technical actuarial report.

---

## 🔁 Reproducibility

The project is designed so the analysis can be reproduced entirely from the R script — no static, hand-uploaded results.

```
Run R Script
     ↓
Data Processed
     ↓
Analysis Performed
     ↓
Figures Generated  →  figures/
     ↓
Numerical Outputs Generated  →  output/
     ↓
Report Results Reproduced
```

To run:

```r
# From the repository root (working directory must be the repo root)
install.packages(c("tidyverse", "caret", "pROC", "survival", "survminer",
                    "corrplot", "broom", "ResourceSelection", "car",
                    "gridExtra", "scales"))

source("Policy_Persistency_Modelling.R")
```

The dataset is provided together with documentation describing its synthetic nature and modelling assumptions.

---

## 🛠️ Tools & Technologies

**Language:** R
**Environment:** RStudio

**Key packages:**
- `tidyverse`
- `caret`
- `pROC`
- `broom`
- `ResourceSelection`
- `car`
- `survival`
- `survminer`
- `corrplot`
- `gridExtra`
- `scales`

The project combines statistical modelling, actuarial reasoning, exploratory analysis, and business interpretation, rather than treating predictive performance as the sole objective.

---

## 📄 Technical Report

For the full methodology, analysis, and interpretation, see:

**[`Actuarial_Policy_Persistency_Modelling.pdf`](report/Actuarial_Policy_Persistency_Modelling.pdf)**

---

## 🎯 Conclusion

This project demonstrates an end-to-end approach to life insurance persistency modelling, from data preparation and actuarial assumptions through predictive modelling, validation, and risk segmentation.

The key conclusion is **not** that the model can perfectly predict lapse.

Rather, the analysis demonstrates how an interpretable statistical model can provide useful relative risk ranking, while also showing why rare-event classification, calibration, model limitations, and the distinction between prediction and causality matter in actuarial applications.

The project therefore focuses on analytical judgement as much as technical implementation:

- Selecting appropriate features
- Evaluating model complexity
- Balancing predictive performance with interpretability
- Validating performance using suitable metrics
- Translating predictions into potential actuarial applications
- Clearly defining what the data and model cannot support

---

## 👤 Author

**Haziq Nazri**
Actuarial Science Graduate · Actuarial Analytics · R · Statistical Modelling
