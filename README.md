# Actuarial Policy Persistency Modelling
An end-to-end actuarial analytics project investigating life insurance policy persistency and lapse risk using R.

The project develops an interpretable logistic regression framework to assess relative lapse risk, evaluates model performance under a highly imbalanced outcome, and demonstrates how predicted risk can support policyholder segmentation and retention prioritisation.

Important: The lapse outcome in this project is synthetic, generated from predefined modelling assumptions. The results therefore demonstrate an actuarial modelling workflow and the ability of the models to recover relationships embedded in the simulated portfolio. They should not be interpreted as empirical evidence of real-world lapse behaviour.

⸻

Project Overview

Policy persistency is an important consideration in life insurance because policyholder behaviour can affect expected future premium income, cash flows, valuation assumptions and retention strategy.

This project develops a complete workflow for identifying policyholders with relatively higher predicted lapse risk. The analysis covers:

* Data quality assessment and preparation
* Actuarial assumptions and feature engineering
* Exploratory data analysis
* Policy persistency analysis
* Predictive modelling
* Model validation
* Risk segmentation
* Actuarial and business interpretation

The project was designed as a portfolio exercise to demonstrate not only technical modelling ability, but also the actuarial judgement required to interpret model results appropriately.

Central Question

Can policyholder characteristics be used to rank policies by relative lapse risk in a way that could support actuarial analysis and targeted retention activity?

A key principle throughout the analysis is the distinction between prediction and causality. A characteristic associated with higher predicted lapse risk does not imply that changing that characteristic would cause lapse behaviour to change.

⸻

Business Problem

For an insurer, understanding which policies are relatively more likely to lapse could potentially support:

* Identification of segments with higher lapse incidence
* Prioritisation of retention resources
* Monitoring of policy persistency
* Experience analysis and assumption development
* Actuarial and business planning

However, lapse prediction is a rare-event classification problem.

When the lapse rate is low, a model can achieve high overall accuracy simply by predicting almost every policy as active. Consequently, this project places greater emphasis on:

Discrimination → Calibration → Risk Ranking → Lift

rather than accuracy alone.

⸻

Dataset

The dataset contains 10,000 synthetic life insurance policyholder records:

Portfolio characteristic	Value
Active policies	9,646
Lapsed policies	354
Overall lapse rate	3.54%

The dataset contains policyholder and policy characteristics across demographic, behavioural, health, product and financial variables.

Data Preparation

The modelling workflow includes:

* Missing-value assessment
* Duplicate checks
* Variable-type and category assessment
* Identification and treatment of identifier variables
* Exclusion of unsuitable free-text/high-cardinality variables
* Transformation of policy start information into policy duration
* Construction of a premium-to-income ratio as an affordability-related feature

The lapse target is generated synthetically using predefined assumptions. This limitation is explicitly considered throughout the analysis.

⸻

Methodology

The project follows an end-to-end actuarial modelling workflow.

1. Data Quality & Preparation

The dataset is assessed for completeness, duplication, variable structure and suitability for modelling.

Identifiers and unsuitable variables are excluded, while policy duration and other modelling features are prepared for subsequent analysis.

2. Actuarial Assumptions & Feature Engineering

The synthetic lapse outcome is constructed from predefined relationships between policyholder characteristics and lapse risk.

Candidate representations are evaluated rather than assuming that additional engineered variables automatically improve the model.

Examples include:

* Age representation
* Monthly premium versus premium-to-income ratio
* Policy duration
* Categorical policyholder characteristics

This allows feature-engineering decisions to be considered using both statistical evidence and actuarial interpretability.

3. Exploratory Data Analysis

Exploratory analysis examines lapse behaviour across demographic, policy, behavioural and financial characteristics.

The objective is not simply to produce descriptive charts, but to identify patterns that can inform modelling decisions and actuarial interpretation.

4. Policy Persistency Analysis

Policy duration and persistency are examined to investigate how lapse behaviour varies across the policy lifecycle.

An illustrative Kaplan–Meier analysis is also used. Because the dataset does not contain genuinely observed, independently recorded lapse event times, the survival analysis is interpreted cautiously and is not treated as evidence of real-world survival behaviour.

5. Predictive Modelling

Four logistic regression specifications are developed:

1. Null / Baseline Model
2. Raw-Variable Model
3. Business-Engineered Model
4. Stepwise AIC-Simplified Model

The final model prioritises a balance between predictive performance, parsimony and interpretability.

The final specification contains:

* Age group
* Smoking status
* Policy type
* Premium-to-income ratio

6. Model Validation

The final model is evaluated using multiple complementary diagnostics:

* Threshold-based classification performance
* ROC / AUC discrimination
* 10-fold cross-validation
* Calibration
* Multicollinearity diagnostics
* Risk ranking and lift

Because lapse is a rare outcome, the conventional 0.50 probability threshold is not treated as an appropriate measure of model usefulness.

7. Risk Segmentation

Predicted lapse probabilities are used to rank policies into risk deciles.

Observed lapse rates and lift are then examined across the risk segments to assess whether the model concentrates a greater proportion of observed lapses within higher-risk groups.

This reframes the model from a simple binary classifier into a potential risk-prioritisation tool.

⸻

Key Findings

Model Development

The modelling progression demonstrates that a more complex feature set does not automatically produce a materially better predictive model.

The final stepwise model achieves the lowest AIC while using substantially fewer parameters than the raw-variable and business-engineered models.

Model	AIC	Parameters	Test AUC
Null / Baseline	2145.9	1	0.500
Raw Variables	2115.4	20	0.643
Business Engineered	2119.2	22	0.635
Stepwise — Final	2103.8	11	0.639

The final model therefore provides a more parsimonious and interpretable specification while retaining broadly similar predictive discrimination to the larger models.

⸻

Model Performance

The final model achieved:

Metric	Result
Test AUC	0.639
10-fold Cross-Validation ROC	≈ 0.620
Hosmer–Lemeshow p-value	≈ 0.68
Maximum VIF	≈ 1.3

The AUC indicates modest discrimination. The model should therefore be interpreted as a risk-ranking framework rather than a highly accurate individual lapse prediction system.

The low lapse incidence also makes accuracy misleading. At the default 0.50 threshold, the model achieves 96.5% accuracy but identifies none of the observed lapses.

Using the illustrative Youden threshold of approximately 0.04 gives:

* Sensitivity: 0.623
* Specificity: 0.604

This demonstrates why threshold selection and appropriate evaluation metrics matter for rare-event actuarial modelling.

⸻

Risk Ranking

The model demonstrates useful separation in relative risk.

The highest-risk decile has:

* 6.35% observed lapse rate
* 3.53% portfolio-average lapse rate
* 1.80× lift

The top 20% predicted-risk segment has an observed lapse rate of approximately 5.8%, compared with approximately 3.5% for the overall portfolio.

This indicates that the model can concentrate a disproportionate share of observed lapses within higher-risk segments.

However, most policyholders within even the highest-risk segment do not lapse. Predicted risk should therefore be interpreted as a tool for prioritisation rather than certainty.

⸻

Actuarial Interpretation

The final model identifies associations between lapse risk and several policyholder characteristics.

Relative to the reference categories:

* Older age groups generally have lower estimated odds of lapse.
* Current smokers have higher estimated odds of lapse.
* Universal Life and Whole Life policies have lower estimated odds of lapse relative to Term Life.
* A higher premium-to-income ratio is associated with higher estimated lapse odds.

These relationships describe model associations within the synthetic dataset and should not be interpreted as causal effects or empirical insurer experience.

⸻

Potential Actuarial Application

The modelling framework could support a conceptual retention workflow:

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

In a real insurance setting, predicted lapse risk could potentially be combined with:

* Policy value
* Customer characteristics
* Intervention cost
* Expected benefit
* Business constraints

to determine where retention resources could be most effectively targeted.

Importantly, predictive modelling alone does not establish that a particular retention intervention will reduce lapse. Intervention testing would be required before making that conclusion.

⸻

IFRS 17 Relevance

Persistency assumptions can influence projected future insurance cash flows and may therefore be relevant to actuarial valuation processes.

This project considers that relevance conceptually, but does not calculate:

* IFRS 17 insurance liabilities
* Contractual Service Margin (CSM)
* Loss components

The IFRS 17 discussion is therefore conceptual rather than a financial reporting calculation.

⸻

Limitations

Several limitations are important when interpreting the results.

Synthetic Outcome

The lapse target is simulated from predefined assumptions rather than derived from observed insurer experience.

The analysis therefore demonstrates whether the modelling workflow can recover relationships embedded in the simulated portfolio rather than discovering empirical lapse drivers.

Modest Predictive Performance

A test AUC of approximately 0.639 represents modest discrimination.

The model is more appropriately interpreted as a risk-ranking framework than as a highly accurate individual prediction system.

Class Imbalance

Only approximately 3.5% of policies lapse.

Consequently, conventional accuracy can give a misleading impression of performance, particularly when using the default 0.50 classification threshold.

Survival-Analysis Limitation

The Kaplan–Meier analysis is illustrative because genuinely observed, independently recorded lapse event times are unavailable.

Prediction ≠ Causality

Associations identified by the model should not be interpreted as causal relationships.

A higher estimated lapse risk for a particular characteristic does not establish that changing that characteristic would reduce lapse.

Generalisability & Deployment

A real insurer would require:

* Observed experience data
* Temporal and external validation
* Appropriate model governance
* Model-drift monitoring
* Fairness assessment
* Business-impact assessment

before any production deployment.

⸻

Future Development

Potential extensions include:

* Genuine event-time survival modelling
* Dynamic policyholder covariates
* Nonlinear model benchmarks
* Interaction modelling
* Uplift / causal modelling
* External validation
* More robust temporal validation
* Cost-sensitive retention optimisation

These extensions would help move the analysis from a portfolio modelling exercise towards a more realistic production-oriented persistency framework.

⸻

Repository Structure

actuarial-policy-persistency/
│
├── README.md
│
├── data/
│   └── life_insurance_retention_dataset_full.csv
│
├── R/
│   └── policy_persistency_modelling.R
│
├── figures/
│   ├── 01_portfolio_lapse_rate.png
│   ├── 02_lapse_by_age_group.png
│   ├── ...
│   └── 13_risk_segmentation.png
│
├── output/
│   ├── model_comparison.csv
│   ├── final_model_coefficients.csv
│   ├── validation_results.csv
│   └── risk_segmentation.csv
│
└── report/
    └── Actuarial_Policy_Persistency_Modelling.pdf

⸻

Repository Contents

data/

Contains the synthetic life insurance dataset used for the analysis.

R/

Contains the complete R modelling workflow, from data preparation through exploratory analysis, modelling, validation and output generation.

figures/

Contains the analytical figures generated by the R workflow.

output/

Contains the key numerical outputs:

* Model comparison
* Final model coefficients
* Validation results
* Risk segmentation

report/

Contains the full technical actuarial report.

⸻

Reproducibility

The project is designed so that the analysis can be reproduced from the R script.

Run R Script
     ↓
Data Processed
     ↓
Analysis Performed
     ↓
Figures Generated
     ↓
Numerical Outputs Generated
     ↓
Report Results Reproduced

The dataset is provided together with documentation describing its synthetic nature and modelling assumptions.

⸻

Tools & Technologies

Language

* R

Environment

* RStudio

Key Packages

* tidyverse
* ggplot2
* dplyr
* broom
* pROC
* ResourceSelection
* car
* survival
* survminer
* gridExtra

The project combines statistical modelling, actuarial reasoning, exploratory analysis and business interpretation, rather than treating predictive performance as the sole objective.

⸻

Conclusion

This project demonstrates an end-to-end approach to life insurance persistency modelling, from data preparation and actuarial assumptions through to predictive modelling, validation and risk segmentation.

The key conclusion is not that the model can perfectly predict lapse.

Instead, the analysis demonstrates how an interpretable statistical model can provide useful relative risk ranking, while also showing why rare-event classification, calibration, model limitations and the distinction between prediction and causality matter in actuarial applications.

The project therefore focuses on analytical judgement as much as technical implementation:

* Selecting appropriate features
* Evaluating model complexity
* Balancing predictive performance with interpretability
* Validating model behaviour using multiple diagnostics
* Translating predictions into potential actuarial applications
* Clearly defining what the data and model cannot support

⸻

Author
Haziq Nazri
Actuarial Science Graduate | Actuarial Analytics | R | Statistical Modelling
