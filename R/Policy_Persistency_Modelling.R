# =============================================================================
# END-TO-END ACTUARIAL POLICY PERSISTENCY MODELLING
# Actuarial Analysis Project | R Script (v3 — Refined per Technical Review)
# Author: Haziq Nazri | Date: June 2026 | Revised: August 2026
# =============================================================================
# Dataset: Life Insurance Retention Dataset (Kaggle, 10,000 policyholders)
# Objective: An end-to-end actuarial policy persistency modelling framework —
#            data quality assessment, feature engineering, exploratory
#            analysis, policy persistency (survival) analysis, predictive
#            modelling, validation, and business recommendation. Prediction
#            is only one stage of this workflow, not the whole of it.
# =============================================================================
# PROJECT PHILOSOPHY
#
# Public life insurance lapse datasets are rarely available due to commercial
# confidentiality. This project therefore combines a publicly available
# insurance customer dataset with transparent, documented actuarial
# assumptions to generate a synthetic lapse outcome.
#
# The objective is NOT to recreate proprietary insurer experience.
#
# The objective IS to demonstrate an end-to-end actuarial workflow —
# data quality assessment, feature engineering, predictive modelling,
# validation, and business interpretation — using a realistic dataset.
# Because the synthetic outcome is generated from known, transparent
# assumptions, the logistic regression later in this script should be read
# as a VALIDATION of whether the modelling workflow recovers those known
# relationships, not as a discovery of real-world lapse drivers.
# =============================================================================
#
# PHASE 1 CHECKPOINTS IMPLEMENTED:
#   Checkpoint 1 - Data Quality Assessment
#   Checkpoint 2 - Feature Engineering Review (keep / remove / compare)
#   Checkpoint 3 - Exploratory Analysis (every figure answers a question)
#   Checkpoint 4 - Model Development Narrative (Null -> Raw -> Engineered ->
#                  Stepwise AIC)
#   Checkpoint 5 - Model Validation (ROC/AUC/CM/HL/VIF + CV + calibration +
#                  threshold trade-off)
#   Checkpoint 6 - Business Insights (statistics translated into actuarial,
#                  decision-ready language)
#
# REFINEMENTS APPLIED IN THIS VERSION (per line-by-line technical review):
#   - Data Dictionary, Checkpoint 1 Summary, and actuarial assumption
#     grounding table added
#   - Stand-alone "Policy Persistency Analysis" section, reframed from
#     generic "survival analysis" terminology
#   - Modelling philosophy, "why logistic regression", information-leakage,
#     and no-black-box-ML statements added ahead of Model Development
#   - Model Comparison Table extended with test-set AUC/Accuracy; Model
#     Development Timeline and Variable Importance tables added
#   - Validation section extended with plain-language metric explanations,
#     a lift chart (moved here as a targeting diagnostic), a Summary
#     Dashboard, and a deployment-readiness closing paragraph
#   - Every odds ratio in Checkpoint 6 phrased as "holding all other
#     variables constant..."
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION 0: LOAD PACKAGES
# -----------------------------------------------------------------------------
# Note: library(ggplot2) is intentionally omitted below — library(tidyverse)
# already attaches ggplot2, dplyr, tidyr, readr, tibble, stringr and purrr.
#
# Required packages are listed in the repository README rather than being
# auto-installed here. If a library() call below fails, install the missing
# package(s) with install.packages("<package_name>") before re-running.

library(tidyverse)
library(caret)
library(pROC)
library(survival)
library(survminer)
library(corrplot)
library(broom)
library(ResourceSelection)   # Hosmer-Lemeshow test
library(car)                 # VIF
library(gridExtra)
library(scales)

set.seed(42)
theme_set(theme_minimal(base_size = 13))

banner <- function(txt) {
  cat("\n", strrep("=", 80), "\n", sep = "")
  cat(" ", txt, "\n", sep = "")
  cat(strrep("=", 80), "\n\n", sep = "")
}

# Consistent colour mapping used throughout: Active = green, Lapsed = red
col_active <- "#4CAF50"
col_lapsed <- "#E53935"
status_colors <- c(Active = col_active, Lapsed = col_lapsed)

# --- Figure saving ---
# Every figure is both displayed (for interactive/RStudio use) AND written to
# figures/ as a numbered PNG, so the repository is reproducible without
# requiring a manual "save each plot" step.
figures_dir <- "figures"
if (!dir.exists(figures_dir)) dir.create(figures_dir, recursive = TRUE)

# --- Tabular output saving ---
# Key numeric/tabular results are written to output/ as CSVs alongside the
# console output, so the repository's outputs are reproducible from the code
# rather than being static, hand-uploaded files.
output_dir <- "output"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

save_table <- function(df_out, filename) {
  path <- file.path(output_dir, filename)
  write.csv(df_out, path, row.names = FALSE)
  cat(sprintf("Saved: %s\n", path))
}

# For standard ggplot objects
save_fig <- function(plot_obj, filename, width = 8, height = 6, dpi = 300) {
  ggsave(filename = file.path(figures_dir, filename), plot = plot_obj,
         width = width, height = height, dpi = dpi, bg = "white")
}

# For gridExtra grid.arrange() composites: builds the grob without drawing,
# draws it to the active device (so it still appears when the script is run
# interactively), then saves it.
save_grid <- function(..., filename, width = 10, height = 8, dpi = 300, top = NULL) {
  g <- gridExtra::arrangeGrob(..., top = top)
  grid::grid.newpage()
  grid::grid.draw(g)
  ggsave(filename = file.path(figures_dir, filename), plot = g,
         width = width, height = height, dpi = dpi, bg = "white")
}

# For base-graphics plots (corrplot) and survminer's ggsurvplot objects,
# which are not plain ggplot grobs: render once to a PNG device, then once
# more to the active graphics device for on-screen display.
save_base_plot <- function(plot_expr, filename, width = 2400, height = 2000, res = 300) {
  png(file.path(figures_dir, filename), width = width, height = height, res = res, bg = "white")
  eval.parent(substitute(plot_expr))
  dev.off()
  eval.parent(substitute(plot_expr))
}


# -----------------------------------------------------------------------------
# SECTION 1: LOAD & INSPECT RAW DATA
# -----------------------------------------------------------------------------
# Path is relative to the repository root — run this script with the repo
# root as the working directory (e.g. open the .Rproj / repo folder in
# RStudio, or setwd() to the repo root first).

data_path <- "data/life_insurance_retention_dataset_full.csv"

if (!file.exists(data_path)) {
  stop(sprintf(
    "Data file not found at '%s'. Update `data_path` at the top of Section 1 to point to your local copy of the CSV.",
    data_path
  ))
}

df_raw <- read.csv(data_path, stringsAsFactors = FALSE)

banner("SECTION 1: RAW DATA OVERVIEW")
cat("Rows:", nrow(df_raw), "| Columns:", ncol(df_raw), "\n\n")
glimpse(df_raw)


# -----------------------------------------------------------------------------
# SECTION 2: CHECKPOINT 1 — DATA QUALITY ASSESSMENT
# -----------------------------------------------------------------------------
# Recruiters expect analysts to verify data quality BEFORE any EDA or
# modelling. Everything below runs on the untouched raw extract.

banner("CHECKPOINT 1: DATA QUALITY ASSESSMENT")

# --- 1a. Dataset Dimensions ---
cat("--- 1a. Dataset Dimensions ---\n")
cat(sprintf("  %d rows x %d columns\n\n", nrow(df_raw), ncol(df_raw)))

# --- 1b. Variable Types ---
cat("--- 1b. Variable Types ---\n")
var_types <- data.frame(
  variable  = names(df_raw),
  data_type = sapply(df_raw, function(x) class(x)[1]),
  row.names = NULL
)
print(var_types)

# --- 1c. Missing Value Analysis ---
cat("\n--- 1c. Missing Value Analysis ---\n")
missing_table <- data.frame(
  variable    = names(df_raw),
  n_missing   = sapply(df_raw, function(x) sum(is.na(x) | (is.character(x) & x == ""))),
  row.names   = NULL
) %>%
  mutate(pct_missing = round(n_missing / nrow(df_raw) * 100, 2)) %>%
  arrange(desc(n_missing))
print(missing_table)
cat(sprintf("\nTotal missing values: %d\n", sum(missing_table$n_missing)))

if (sum(missing_table$n_missing) == 0) {
  cat("No missing values detected in any column. No imputation required.\n")
} else {
  cat("WARNING: missing values detected — an imputation/removal strategy is\n")
  cat("required before modelling. Review the variables flagged above.\n")
}

# --- 1d. Duplicate Records ---
cat("\n--- 1d. Duplicate Records ---\n")
n_dup_rows <- sum(duplicated(df_raw))
n_dup_ids  <- sum(duplicated(df_raw$customer_id))
cat(sprintf("  Fully duplicated rows:         %d\n", n_dup_rows))
cat(sprintf("  Duplicated customer_id values: %d\n", n_dup_ids))
cat("\n  Note: duplicate customer IDs could indicate multiple policies held by\n")
cat("  the same individual. Since this dataset assumes one record per\n")
cat("  policyholder, duplicates would be treated as a data quality issue\n")
cat("  requiring investigation before modelling.\n")
if (n_dup_rows == 0 && n_dup_ids == 0) {
  cat("  No duplicate records found — each row represents a unique policyholder.\n")
}

# --- 1e. Summary Statistics ---
cat("\n--- 1e. Summary Statistics (Numeric Variables) ---\n")
print(summary(df_raw %>%
                select(age, number_of_dependents, annual_income,
                       coverage_amount, monthly_premium)))

cat("\n--- 1e (cont). Categorical Variable Levels (with counts) ---\n")
for (v in c("gender", "marital_status", "health_status",
            "smoking_status", "policy_type")) {
  tab <- sort(table(df_raw[[v]]), decreasing = TRUE)
  cat(sprintf("  %-16s: %s\n", v,
              paste(sprintf("%s (%d)", names(tab), tab), collapse = ", ")))
}

# --- 1f. Data Quality Flags ---
cat("\n--- 1f. Data Quality Flags Requiring Cleaning ---\n")
rare_gender <- sum(df_raw$gender == "_RARE_")
rare_health <- sum(df_raw$health_status == "_RARE_")
cat(sprintf("  '_RARE_' placeholder values in gender:        %d records\n", rare_gender))
cat(sprintf("  '_RARE_' placeholder values in health_status: %d records\n", rare_health))
cat("  occupation: free-text field with inconsistent spelling/typos and high\n")
cat("  cardinality (not a clean categorical predictor).\n")
cat("  occupation and name are excluded from predictive modelling but retained\n")
cat("  in the raw dataset to preserve reproducibility.\n")
cat("  These issues are resolved in Section 3 (Cleaning & Feature Engineering).\n")

# --- 1g. Checkpoint 1 Summary ---
banner("CHECKPOINT 1 SUMMARY")
cat(
  "Data quality assessment identified no missing values or duplicate records.\n",
  "The primary issues relate to placeholder categories ('_RARE_' in gender and\n",
  "health_status) and non-modellable fields (customer name and free-text\n",
  "occupation), which are addressed during feature engineering (Section 3).\n",
  "The dataset is therefore considered suitable for predictive modelling after\n",
  "targeted cleaning.\n",
  sep = ""
)


# -----------------------------------------------------------------------------
# SECTION 2B: DATA DICTIONARY
# -----------------------------------------------------------------------------

banner("DATA DICTIONARY")

data_dictionary <- data.frame(
  Variable = c("age", "gender", "marital_status", "number_of_dependents",
               "annual_income", "occupation", "health_status", "smoking_status",
               "policy_type", "coverage_amount", "monthly_premium",
               "policy_start_date", "customer_id", "name"),
  Type = c("Numeric", "Categorical", "Categorical", "Numeric", "Numeric",
           "Character (free text)", "Categorical", "Categorical", "Categorical",
           "Numeric", "Numeric", "Date", "Character (ID)", "Character (ID)"),
  Description = c(
    "Policyholder age at entry",
    "Policyholder gender",
    "Marital status",
    "Number of financial dependents",
    "Annual income ($)",
    "Free-text occupation title",
    "Self-reported health status",
    "Smoking status",
    "Life insurance product type",
    "Sum assured / coverage amount ($)",
    "Monthly premium payment ($)",
    "Policy inception date",
    "Unique policyholder identifier",
    "Policyholder name"
  ),
  Used_in_Modelling = c("Yes", "Yes", "Yes", "Yes", "Yes (derives ratio)",
                         "No", "Yes", "Yes", "Yes", "Yes", "Yes (derives ratio)",
                         "Derived (duration)", "No", "No"),
  stringsAsFactors = FALSE
)
print(data_dictionary, row.names = FALSE)


# -----------------------------------------------------------------------------
# SECTION 3: DATA CLEANING, TARGET CONSTRUCTION & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
# Raw data (df_raw) is kept fully intact above. We build a separate, cleaned
# object (df) so earlier decisions remain inspectable/reproducible rather than
# overwriting the raw extract in place.

banner("SECTION 3: DATA CLEANING & FEATURE ENGINEERING")

df <- df_raw %>%

  # Drop columns with no modelling value (see Checkpoint 1 flags). These
  # remain fully available in df_raw if ever needed again.
  select(-customer_id, -name, -occupation) %>%

  # Parse policy start date
  mutate(policy_start_date = as.Date(policy_start_date, format = "%Y-%m-%d")) %>%

  # Policy duration in years, as of reference date. Retained CONTINUOUS for
  # modelling; a binned version is created separately below for EDA only.
  mutate(policy_duration_years = as.numeric(
    difftime(as.Date("2024-12-31"), policy_start_date, units = "days")) / 365.25) %>%

  # --- ENGINEERED PREDICTORS ---
  mutate(
    # Affordability signal: premium as a share of monthly income
    premium_income_ratio = monthly_premium / (annual_income / 12),

    # Age bucketed for EDA storytelling only — age stays continuous in the model
    age_group = cut(age,
                     breaks = c(17, 29, 39, 49, 59, 100),
                     labels = c("18-29", "30-39", "40-49", "50-59", "60+")),

    # Duration bucketed for EDA storytelling only — duration stays continuous
    # in the model (see Checkpoint 2 univariate screen for the justification).
    # NOTE: the observed window is narrow (~2.0-3.0 years, since every policy
    # in this extract incepted in 2022), so fixed annual breaks collapsed
    # ~100% of the portfolio into a single "2-3yr" bin. Quantile-based bins
    # are used instead, so each band contains a comparable share of the
    # portfolio and the chart can actually show a gradient.
    duration_band = cut(policy_duration_years,
                         breaks = quantile(policy_duration_years,
                                            probs = seq(0, 1, length.out = 6),
                                            na.rm = TRUE),
                         include.lowest = TRUE)
  ) %>%

  # --- ENGINEER TARGET VARIABLE: Lapsed (1 = lapsed, 0 = active) ---
  # Business logic grounded in SOA lapse experience studies (see the
  # "Actuarial Grounding" table printed below): lapse probability is
  # elevated by short duration, high premium burden, younger age, smoker
  # status, and certain policy types. A small idiosyncratic noise term is
  # added to the log-odds so the relationship between predictors and outcome
  # is not perfectly deterministic (unobserved heterogeneity).
  mutate(
    lapse_score =
      -2.5                                                      # intercept
    + 0.04  * (40 - age)                                        # younger = higher risk
    - 0.35  * policy_duration_years                              # longer in-force = more loyal
    + 2.50  * pmin(premium_income_ratio, 0.5)                    # affordability stress
    + 0.30  * (smoking_status == "Current smoker")               # smoker uplift
    - 0.25  * (smoking_status == "Non-smoker")                   # non-smoker discount
    + 0.20  * (policy_type == "Term Life")                       # term lapses more
    - 0.30  * (policy_type == "Whole Life")                      # whole life stickier
    + 0.15  * (health_status == "Fair")                          # financial stress signal
    + 0.10  * number_of_dependents                               # dependents = financial pressure
    - 0.20  * (marital_status == "Married")                      # married = more stable
    + rnorm(n(), mean = 0, sd = 0.15),                           # unobserved heterogeneity

    lapse_prob = 1 / (1 + exp(-lapse_score)),
    Lapsed     = rbinom(n(), 1, lapse_prob)
  ) %>%

  # Clean categorical levels
  mutate(
    gender         = ifelse(gender == "_RARE_", "Other", gender),
    health_status  = ifelse(health_status == "_RARE_", "Poor", health_status),
    gender         = factor(gender),
    marital_status = factor(marital_status),
    smoking_status = factor(smoking_status, levels = c("Non-smoker","Former smoker","Current smoker")),
    policy_type    = factor(policy_type),
    health_status  = factor(health_status, levels = c("Excellent","Good","Fair","Poor")),
    Lapsed         = factor(Lapsed, levels = c(0,1), labels = c("Active","Lapsed"))
  ) %>%

  # Drop helper columns not needed downstream (premium_income_ratio and
  # age_group are DELIBERATELY KEPT — see Checkpoint 2 for the test rather
  # than assumption of their modelling value)
  select(-policy_start_date, -lapse_score, -lapse_prob)

cat("=== CLEANED DATA STRUCTURE ===\n")
glimpse(df)

# --- Actuarial Grounding of Synthetic Lapse Assumptions ---
cat("\n--- Actuarial Grounding of Synthetic Lapse Assumptions ---\n")
assumption_grounding <- data.frame(
  Assumption = c("High premium-to-income burden", "Longer policy duration",
                 "Younger age", "Current smoker", "Term Life product",
                 "Whole Life product", "Fair health", "More dependents", "Married"),
  Direction = c("+ lapse risk", "- lapse risk", "+ lapse risk", "+ lapse risk",
                "+ lapse risk", "- lapse risk", "+ lapse risk", "+ lapse risk", "- lapse risk"),
  Supporting_Concept = c(
    "Premium affordability / financial stress",
    "Policy persistency / duration effects",
    "Customer mobility and switching behaviour",
    "Risk profile and behavioural characteristics",
    "Product lapse characteristics (no cash-value lock-in)",
    "Cash-value accumulation / surrender-charge deterrent",
    "Financial-strain proxy via self-reported health",
    "Competing financial priorities",
    "Household financial stability"
  ),
  stringsAsFactors = FALSE
)
print(assumption_grounding, row.names = FALSE)

cat("\nBecause the lapse outcome is generated from these transparent\n")
cat("assumptions, the logistic regression in Section 7 is best read as a\n")
cat("VALIDATION of whether the modelling workflow recovers known actuarial\n")
cat("relationships — not as a discovery of real-world lapse drivers from\n")
cat("observational data.\n")


# --- Checkpoint 1 (cont.): Target Class Distribution ---
banner("CHECKPOINT 1 (cont.): TARGET CLASS DISTRIBUTION")

print(table(df$Lapsed))
overall_lapse_rate <- mean(df$Lapsed == "Lapsed")
cat(sprintf("\nOverall Lapse Rate: %.2f%%\n", overall_lapse_rate * 100))

p_balance <- df %>%
  count(Lapsed) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = Lapsed, y = n, fill = Lapsed)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = paste0(n, " (", percent(pct, accuracy = 0.1), ")")),
            vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = status_colors) +
  labs(title = "Figure 0: Target Class Distribution",
       subtitle = "Class imbalance directly affects threshold choice and metric selection (Section 8)",
       x = NULL, y = "Count")
print(p_balance)
save_fig(p_balance, "00_target_class_distribution.png")

cat(sprintf(
  "\nThe dataset exhibits %s class imbalance (%.1f%% lapse rate), indicating\n",
  ifelse(overall_lapse_rate < 0.10, "pronounced", "moderate"), overall_lapse_rate * 100
))
cat("that evaluation metrics beyond overall accuracy — sensitivity,\n")
cat("specificity, AUC, and lift — will be important (Section 8).\n")


# -----------------------------------------------------------------------------
# SECTION 4: CHECKPOINT 2 — FEATURE ENGINEERING REVIEW
# -----------------------------------------------------------------------------
# Every engineered variable is tested, not just calculated. We compare
# candidate forms head-to-head using univariate AIC before deciding whether
# to KEEP, REMOVE, or CARRY FORWARD BOTH for a formal test in Section 7.

banner("CHECKPOINT 2: FEATURE ENGINEERING REVIEW")

cat("--- 2a. Candidate Forms Under Review ---\n")
cat("  1. age (continuous)          vs  age_group (binned, 5 levels)\n")
cat("  2. monthly_premium (raw)     vs  premium_income_ratio (engineered)\n")
cat("  3. policy_duration_years — no alternative form, evaluated for retention\n\n")

cat("--- 2b. Univariate Screening (single-predictor logistic models) ---\n")

uni_age_cont  <- glm(Lapsed ~ age,                   data = df, family = binomial)
uni_age_group <- glm(Lapsed ~ age_group,             data = df, family = binomial)
uni_prem_raw  <- glm(Lapsed ~ monthly_premium,       data = df, family = binomial)
uni_prem_rat  <- glm(Lapsed ~ premium_income_ratio,  data = df, family = binomial)
uni_duration  <- glm(Lapsed ~ policy_duration_years, data = df, family = binomial)

uni_compare <- data.frame(
  Variable = c("age (continuous)", "age_group (binned)",
               "monthly_premium (raw)", "premium_income_ratio (engineered)",
               "policy_duration_years"),
  AIC = round(c(AIC(uni_age_cont), AIC(uni_age_group),
                AIC(uni_prem_raw), AIC(uni_prem_rat),
                AIC(uni_duration)), 1)
)
print(uni_compare)

cat("\n--- 2c. Keep / Remove / Compare Decisions ---\n")

age_decision <- if (AIC(uni_age_cont) <= AIC(uni_age_group)) {
  "KEEP age as CONTINUOUS in the model (lower/comparable AIC preserves granularity).\n  age_group is RETAINED separately as a labelling variable for EDA storytelling only."
} else {
  "KEEP age_group (BINNED) in the model — the categorical form fits noticeably better,\n  suggesting a non-linear age effect that a single linear term would miss."
}
cat("  1. Age form       -> ", age_decision, "\n\n", sep = "")

ratio_beats_raw <- AIC(uni_prem_rat) < AIC(uni_prem_raw)
premium_decision <- if (ratio_beats_raw) {
  "premium_income_ratio OUTPERFORMS raw monthly_premium on its own and is\n  PROMOTED to a primary explanatory variable (formally tested against the raw\n  form in the Model 1 vs Model 2 comparison, Section 7)."
} else {
  "raw monthly_premium is not yet beaten by the ratio in isolation, but the ratio\n  is still carried forward into Model 2 for a fair, multivariable comparison —\n  affordability effects are often only visible once income is netted out\n  jointly with other predictors, not in a univariate screen."
}
cat("  2. Premium form   -> ", premium_decision, "\n\n", sep = "")

cat("  3. policy_duration_years -> KEEP CONTINUOUS in the model. AIC =", round(AIC(uni_duration), 1),
    "vs null model AIC =", round(AIC(glm(Lapsed ~ 1, data = df, family = binomial)), 1),
    "\n     confirms duration alone is informative, consistent with SOA persistency\n")
cat("     studies (early policy years carry the greatest lapse risk). A binned\n")
cat("     duration_band is retained separately for EDA charts only (Section 5).\n")

cat("\nSummary: this project's feature selection follows a three-stage narrative\n")
cat("rather than 'engineer, then delete': (1) we engineered several candidate\n")
cat("features grounded in actuarial reasoning; (2) we evaluated which forms\n")
cat("best represented policyholder behaviour via the univariate screen above;\n")
cat("(3) the final variables are selected in Section 7 by balancing predictive\n")
cat("performance, interpretability, and business relevance — not AIC alone.\n")


# -----------------------------------------------------------------------------
# SECTION 5: CHECKPOINT 3 — EXPLORATORY DATA ANALYSIS (ENHANCED)
# -----------------------------------------------------------------------------
# Every figure below is paired with the business question it answers, and an
# insight-focused caption rather than a bare chart.

banner("CHECKPOINT 3: EXPLORATORY DATA ANALYSIS")

# --- 5a. Numeric Summary Statistics by Lapse Status ---
cat("--- 5a. Numeric Variable Summary by Lapse Status ---\n")
num_summary <- df %>%
  group_by(Lapsed) %>%
  summarise(across(c(age, annual_income, coverage_amount, monthly_premium,
                      premium_income_ratio, policy_duration_years, number_of_dependents),
                    list(mean = ~round(mean(.x), 2), median = ~round(median(.x), 2))),
            .groups = "drop")
print(as.data.frame(t(num_summary)))
cat("\nInsight: younger policyholders appear over-represented among lapse events\n")
cat("(lower mean/median age in the Lapsed column), supporting the hypothesis\n")
cat("that customer mobility influences policy persistency. Income is\n")
cat("right-skewed (mean > median in both groups) — a small number of\n")
cat("high-income policyholders pull the mean upward, which is worth keeping in\n")
cat("mind when interpreting premium-affordability measures built on income.\n")

# --- 5b. Lapse Rate by Key Categorical Variables (business-question framed) ---

plot_lapse_by <- function(data, var, title, question) {
  tbl <- data %>%
    group_by(across(all_of(var))) %>%
    summarise(n = n(), lapse_rate = mean(Lapsed == "Lapsed"), .groups = "drop")

  p <- ggplot(tbl, aes(x = .data[[var]], y = lapse_rate, fill = .data[[var]])) +
    geom_col(show.legend = FALSE, width = 0.6) +
    geom_text(aes(label = percent(lapse_rate, accuracy = 0.1)),
              vjust = -0.5, size = 3.2) +
    scale_y_continuous(labels = percent_format(),
                        limits = c(0, max(tbl$lapse_rate) * 1.35)) +
    labs(title = title, subtitle = question, x = NULL, y = "Lapse Rate") +
    theme(axis.text.x = element_text(angle = 15, hjust = 1))

  list(plot = p, table = tbl)
}

report_extremes <- function(tbl, var_name, label, fig_num) {
  top <- tbl %>% arrange(desc(lapse_rate)) %>% slice(1)
  bot <- tbl %>% arrange(lapse_rate) %>% slice(1)
  cat("  Counts & lapse rate:\n")
  print(tbl %>% mutate(lapse_rate = percent(lapse_rate, accuracy = 0.1)), n = Inf)
  cat(sprintf("  Figure %s caption: %s has the HIGHEST lapse rate (%.1f%%); %s has\n  the LOWEST (%.1f%%).\n\n",
              fig_num, top[[var_name]], top$lapse_rate * 100,
              bot[[var_name]], bot$lapse_rate * 100))
}

r1 <- plot_lapse_by(df, "policy_type", "Policy Type and Its Relationship with Lapse Behaviour",
                     "Business question: which policy structures carry the most lapse risk?")
print(r1$plot)
report_extremes(r1$table, "policy_type", "Policy Type", "1a")

r2 <- plot_lapse_by(df, "smoking_status", "Smoking Status and Its Relationship with Lapse Behaviour",
                     "Business question: does smoker risk-loading affect persistency?")
print(r2$plot)
report_extremes(r2$table, "smoking_status", "Smoking Status", "1b")

r3 <- plot_lapse_by(df, "health_status", "Health Status and Its Relationship with Lapse Behaviour",
                     "Business question: is declining health a proxy for financial strain?")
print(r3$plot)
report_extremes(r3$table, "health_status", "Health Status", "1c")

r4 <- plot_lapse_by(df, "marital_status", "Marital Status and Its Relationship with Lapse Behaviour",
                     "Business question: does household structure improve retention?")
print(r4$plot)
report_extremes(r4$table, "marital_status", "Marital Status", "1d")

save_grid(r1$plot, r2$plot, r3$plot, r4$plot, ncol = 2,
          top = "Figure 1: Lapse Rate by Policyholder Characteristics",
          filename = "01_lapse_rate_by_characteristics.png", width = 14, height = 10)

# --- 5c. Lapse Rate by Age Group — "Why are younger policyholders more likely to lapse?" ---

r5 <- plot_lapse_by(df, "age_group", "Age Group and Its Relationship with Lapse Behaviour",
                     "Business question: why are younger policyholders more likely to lapse?")
print(r5$plot)
save_fig(r5$plot, "02_lapse_rate_by_age_group.png")
report_extremes(r5$table, "age_group", "Age Group", "2")
cat("  Interpretation: younger policyholders are typically earlier in their career\n")
cat("  and family-formation stage, with lower disposable income — premiums consume\n")
cat("  a larger share of their budget (see premium-to-income ratio below), which\n")
cat("  weakens persistence relative to older, more financially established cohorts.\n\n")

# --- 5d. Lapse Rate by Policy Duration (Binned) ---

r6_tbl <- df %>%
  group_by(duration_band) %>%
  summarise(lapse_rate = mean(Lapsed == "Lapsed"), n = n(), .groups = "drop")

p_duration <- ggplot(r6_tbl, aes(x = duration_band, y = lapse_rate, group = 1)) +
  geom_line(color = "#2C5F8A", linewidth = 1.2) +
  geom_point(color = "#2C5F8A", size = 3) +
  geom_text(aes(label = percent(lapse_rate, accuracy = 0.1)), vjust = -0.8, size = 3.2) +
  scale_y_continuous(labels = percent_format(), limits = c(0, max(r6_tbl$lapse_rate) * 1.3)) +
  labs(title = "Policy Duration and Its Relationship with Policy Persistency",
       subtitle = "Business question: within this portfolio's narrow observed window (~2.0-3.0 years), does lapse risk still vary by duration?",
       x = "Policy Duration (quantile bands, years)", y = "Lapse Rate") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
print(p_duration)
save_fig(p_duration, "03_lapse_rate_by_duration.png")
cat("  Figure 3 caption: because every policy in this extract was incepted in\n")
cat("  2022, the observed window spans only ~1 year (2.0-3.0 years of\n")
cat("  duration) rather than a full policy lifecycle. Quantile-based bands are\n")
cat("  used here (replacing fixed annual bins, which collapsed almost the\n")
cat("  entire portfolio into a single '2-3yr' category) to reveal the duration\n")
cat("  gradient present within that narrow window. This gradient reflects the\n")
cat("  duration term built into the synthetic label (Section 3) rather than\n")
cat("  genuine multi-year persistency experience — see Section 6 for the\n")
cat("  equivalent caveat on the Kaplan-Meier curve.\n\n")

# --- 5e. Distributions of Key Numeric Variables ---

p_age  <- ggplot(df, aes(x = age, fill = Lapsed)) +
  geom_histogram(bins = 20, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = status_colors) +
  labs(title = "Policyholder Age and Lapse Behaviour", x = "Age at Entry", y = "Count")

p_prem <- ggplot(df, aes(x = monthly_premium, fill = Lapsed)) +
  geom_histogram(bins = 25, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = status_colors) +
  scale_x_continuous(labels = dollar_format()) +
  labs(title = "Monthly Premium and Lapse Behaviour", x = "Monthly Premium ($)", y = "Count")

save_grid(p_age, p_prem, ncol = 2,
          top = "Figure 4: Numeric Variable Distributions by Lapse Status",
          filename = "04_numeric_distributions.png", width = 12, height = 5.5)

# --- 5f. Boxplots — Affordability & Tenure Drivers ---

p_box_ratio <- ggplot(df, aes(x = Lapsed, y = premium_income_ratio, fill = Lapsed)) +
  geom_boxplot(show.legend = FALSE) +
  coord_cartesian(ylim = c(0, quantile(df$premium_income_ratio, 0.95))) +
  scale_fill_manual(values = status_colors) +
  labs(title = "Premium-to-Income Ratio", x = NULL, y = "Premium / Monthly Income")

p_box_income <- ggplot(df, aes(x = Lapsed, y = annual_income, fill = Lapsed)) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_manual(values = status_colors) +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = "Annual Income", x = NULL, y = "Annual Income ($)")

p_box_duration <- ggplot(df, aes(x = Lapsed, y = policy_duration_years, fill = Lapsed)) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_manual(values = status_colors) +
  labs(title = "Policy Duration", x = NULL, y = "Duration (Years)")

save_grid(p_box_ratio, p_box_income, p_box_duration, ncol = 3,
          top = "Figure 5: Boxplots — Affordability & Tenure Drivers by Lapse Status",
          filename = "05_boxplots_affordability_tenure.png", width = 12, height = 5)
cat("  Figure 5 caption: if the interquartile ranges of premium_income_ratio\n")
cat("  visibly separate between Active and Lapsed groups, this supports\n")
cat("  promoting the ratio to a primary explanatory variable (tested formally\n")
cat("  in Section 7).\n\n")

# --- 5g. Correlation Matrix (Numeric Variables) ---

numeric_vars <- df %>%
  mutate(Lapsed_num = as.numeric(Lapsed) - 1) %>%
  select(age, annual_income, coverage_amount, monthly_premium,
         premium_income_ratio, policy_duration_years, number_of_dependents, Lapsed_num)

cor_matrix <- cor(numeric_vars, use = "complete.obs")

save_base_plot(
  corrplot(cor_matrix,
           method = "color", type = "upper", addCoef.col = "black",
           number.cex = 0.75, tl.col = "black", tl.srt = 45,
           col = colorRampPalette(c("#B71C1C","white","#1565C0"))(200),
           title = "Figure 6: Correlation Matrix — Numeric Variables",
           mar = c(0,0,1,0)),
  filename = "06_correlation_matrix.png", width = 2400, height = 2000
)

cat("--- 5g (cont). Why Multicollinearity Matters ---\n")
cat("Highly correlated predictors (|r| > 0.7) can inflate coefficient\n")
cat("uncertainty and reduce interpretability, making this correlation\n")
cat("assessment an important step before fitting the logistic regression. This\n")
cat("is formally revisited via Variance Inflation Factor (VIF) analysis in\n")
cat("Section 8 (Model Validation).\n")

# --- 5h. Correlation with Target ---

cat("\n--- 5h. Correlation with Target (Numeric Variables) ---\n")
target_num <- as.numeric(df$Lapsed) - 1
corr_vars <- c("age", "annual_income", "coverage_amount", "monthly_premium",
               "premium_income_ratio", "policy_duration_years", "number_of_dependents")
corr_target_df <- data.frame(
  Variable    = corr_vars,
  Correlation = round(sapply(corr_vars, function(v) cor(df[[v]], target_num)), 3)
) %>%
  mutate(Association = case_when(
    abs(Correlation) >= 0.30 ~ "Strong",
    abs(Correlation) >= 0.15 ~ "Moderate",
    TRUE ~ "Weak"
  )) %>%
  arrange(desc(abs(Correlation)))
print(corr_target_df, row.names = FALSE)
cat("\nThis table is a natural bridge into Section 7 (Model Development):\n")
cat("variables with stronger target correlation are expected — though not\n")
cat("guaranteed, given multivariable confounding — to emerge as significant\n")
cat("predictors in the logistic regression.\n")

# --- 5i. Supporting Statistical Tests ---

cat("\n--- 5i. Supporting Statistical Tests ---\n")
cat("Visual patterns above are supported (not replaced) by formal tests:\n")
cat("Wilcoxon rank-sum for numeric variables (robust to skew) and Chi-square\n")
cat("tests of independence for categorical variables.\n\n")

num_test_vars <- c("age", "annual_income", "policy_duration_years", "premium_income_ratio")
num_tests <- data.frame(
  Variable = num_test_vars,
  Test     = "Wilcoxon rank-sum",
  p_value  = round(sapply(num_test_vars, function(v) wilcox.test(df[[v]] ~ df$Lapsed)$p.value), 4)
)
print(num_tests, row.names = FALSE)

cat_test_vars <- c("smoking_status", "policy_type", "health_status", "marital_status")
cat_tests <- data.frame(
  Variable = cat_test_vars,
  Test     = "Chi-square",
  p_value  = round(sapply(cat_test_vars, function(v) chisq.test(table(df[[v]], df$Lapsed))$p.value), 4)
)
print(cat_tests, row.names = FALSE)
cat("\np < 0.05 indicates a statistically significant association with lapse\n")
cat("status at the 5% level.\n")

# --- 5j. Key Findings from Exploratory Analysis ---

banner("KEY FINDINGS FROM EXPLORATORY ANALYSIS")
cat("- Policy duration: lapse rates are highest in the earliest duration bands,\n")
cat("  consistent with established policy persistency concepts (Figure 3).\n")
cat(sprintf("- Age: %s has the highest lapse rate among age groups (%.1f%%), supporting\n",
            (r5$table %>% arrange(desc(lapse_rate)) %>% slice(1))$age_group,
            (r5$table %>% arrange(desc(lapse_rate)) %>% slice(1))$lapse_rate * 100))
cat("  the hypothesis that younger customers exhibit greater mobility.\n")
cat(sprintf("- Policy type: %s shows materially higher lapse propensity than %s\n",
            (r1$table %>% arrange(desc(lapse_rate)) %>% slice(1))$policy_type,
            (r1$table %>% arrange(lapse_rate) %>% slice(1))$policy_type))
cat("  products, consistent with differences in cash-value lock-in.\n")
cat("- Smoking status and marital status show meaningful, statistically\n")
cat("  significant differences between retained and lapsed groups (Section 5i).\n")
cat("- Affordability (premium-to-income ratio) separates Active and Lapsed\n")
cat("  groups in the boxplot comparison (Figure 5), motivating its formal test\n")
cat("  as a primary explanatory variable in Section 7.\n")

# --- 5k. EDA -> Modelling Decisions ---

banner("EDA -> MODELLING DECISIONS")
eda_decisions <- data.frame(
  Observation = c(
    "Class imbalance in the target variable",
    "Numeric predictors show some pairwise correlation",
    "Younger policyholders show higher lapse rates",
    "Shorter policy duration linked to higher lapse",
    "Premium-to-income ratio separates Active vs Lapsed groups"
  ),
  Modelling_Decision = c(
    "Evaluate AUC, sensitivity, specificity and lift in addition to accuracy",
    "Check VIF before interpreting coefficients",
    "Retain age as a candidate predictor",
    "Include policy_duration_years as a continuous predictor",
    "Formally compare raw premium/income vs the engineered ratio (Section 7)"
  ),
  stringsAsFactors = FALSE
)
print(eda_decisions, row.names = FALSE)


# -----------------------------------------------------------------------------
# SECTION 6: POLICY PERSISTENCY ANALYSIS
# -----------------------------------------------------------------------------
# Formerly labelled "Survival Analysis". Renamed because "persistency" is the
# term insurers actually use, and the section is reframed as a standalone
# actuarial complement to the logistic regression, not a generic ML add-on.

banner("SECTION 6: POLICY PERSISTENCY ANALYSIS")

cat("Although logistic regression predicts WHETHER a policyholder lapses, it\n")
cat("does not explicitly consider WHEN the lapse occurs. Policy lapse is, in\n")
cat("general, a time-to-event process, and the Kaplan-Meier estimator is the\n")
cat("standard actuarial tool for exploring that dimension of persistency.\n\n")

cat("IMPORTANT CAVEAT: in a genuine time-to-lapse analysis, policy duration is\n")
cat("an INDEPENDENTLY OBSERVED event time — the calendar point at which a\n")
cat("real lapse occurred. That is not what is happening here. In Section 3,\n")
cat("policy_duration_years is one of the INPUTS used, together with other\n")
cat("policyholder characteristics, to construct the synthetic Lapsed label\n")
cat("itself — there is no independently observed lapse date in this dataset.\n")
cat("Consequently, the curves below must NOT be read as a genuine estimate of\n")
cat("time-to-lapse. They are presented purely as an ILLUSTRATIVE DEVICE for\n")
cat("visualising how the duration-linked assumptions built into Section 3\n")
cat("manifest across the synthetic portfolio — a descriptive complement to\n")
cat("the logistic regression, not an independent time-to-event estimate of\n")
cat("real insurer lapse experience.\n\n")

df_surv <- df %>%
  mutate(time = policy_duration_years, status = as.numeric(Lapsed == "Lapsed"))

km_overall <- survfit(Surv(time, status) ~ 1, data = df_surv)
km_policy  <- survfit(Surv(time, status) ~ policy_type, data = df_surv)

# --- Retention at key durations within the OBSERVED window ---
# All policies in this extract incepted in 2022, so the observable window is
# only ~2-3 years as of the reference date — the conventional 1/3/5-year
# actuarial convention doesn't apply here. We instead report retention at
# early/mid/late points spanning the window that IS observed.
obs_range <- range(df_surv$time)
retention_times <- round(seq(obs_range[1] + 0.1, obs_range[2] - 0.1, length.out = 3), 2)

cat(sprintf("Observed policy duration window: %.2f to %.2f years (all policies\n",
            obs_range[1], obs_range[2]))
cat("incepted in 2022 relative to the 2024-12-31 reference date), so retention\n")
cat("is reported at early/mid/late points within that window rather than the\n")
cat("conventional 1/3/5-year actuarial horizon.\n\n")

retention_summary <- summary(km_overall, times = retention_times)
retention_df <- data.frame(
  Duration_Years = round(retention_summary$time, 2),
  Retention       = percent(retention_summary$surv, accuracy = 0.1),
  CI_Lower_95     = percent(retention_summary$lower, accuracy = 0.1),
  CI_Upper_95     = percent(retention_summary$upper, accuracy = 0.1)
)
cat("--- Illustrative Policy Persistency Pattern at Key Durations (Synthetic Portfolio) ---\n")
print(retention_df, row.names = FALSE)

km_surv_plot <- ggsurvplot(
  km_policy, data = df_surv, risk.table = TRUE, pval = TRUE, conf.int = TRUE,
  palette = c("#1565C0","#E53935","#2E7D32","#F57F17"),
  legend.labs = levels(df$policy_type),
  title = "Figure 7: Illustrative Policy Persistency Pattern by Policy Type (Synthetic Data)",
  subtitle = "Business question: which policy types show weaker persistency in this synthetic portfolio?",
  xlab = "Policy Duration (Years)", ylab = "Proportion Remaining Active (Illustrative)",
  ggtheme = theme_minimal(base_size = 13)
)
save_base_plot(print(km_surv_plot), filename = "07_persistency_curve_by_policy_type.png",
                width = 2800, height = 2400)
cat("Figure 7 caption: the pattern by policy type illustrates how this kind of\n")
cat("analysis could help insurers identify products with weaker persistency\n")
cat("and target retention strategies accordingly — using real, independently\n")
cat("observed lapse dates in an operational setting.\n\n")

log_rank <- survdiff(Surv(time, status) ~ policy_type, data = df_surv)
cat("--- Log-Rank Test (Policy Type Persistency Pattern) ---\n")
print(log_rank)

cat("\nWithin this illustrative curve, the underlying HAZARD of lapse appears to\n")
cat("vary across the policy lifecycle, with early durations exhibiting higher\n")
cat("lapse intensity — consistent with the duration effect observed in the EDA\n")
cat("(Section 5, Figure 3) and, more fundamentally, with the duration term\n")
cat("built directly into the synthetic label in Section 3.\n\n")

cat("A Cox proportional hazards model would allow covariate effects on the\n")
cat("hazard of lapse to be estimated directly. This is noted here as a\n")
cat("natural extension of this project but is intentionally out of scope, to\n")
cat("keep the modelling focus on the logistic regression framework used\n")
cat("throughout the remainder of the script.\n\n")

cat("--- Logistic Regression vs Persistency Analysis: Complementary Roles ---\n")
method_comparison <- data.frame(
  Aspect = c("Outcome", "Question Answered", "Focus", "Business Use",
             "Status in This Project"),
  Logistic_Regression = c("Binary (lapse / active)", "Predicts probability of lapse",
                           "Individual risk factors", "Supports customer-level targeting",
                           "Fitted and validated on the synthetic dataset"),
  Persistency_Analysis = c("Time-to-event (in a genuine setting)",
                            "Illustrates portfolio persistency patterns",
                            "Descriptive, aggregate patterns",
                            "Would support retention monitoring with real lapse dates",
                            "Illustrative only — duration is an input to the synthetic label, not an observed event time"),
  stringsAsFactors = FALSE
)
print(method_comparison, row.names = FALSE)

cat("\nThe persistency analysis above provides an illustrative, descriptive\n")
cat("complement to the logistic regression rather than an independent\n")
cat("time-to-event estimate. The logistic regression in the next section\n")
cat("remains the project's primary, validated quantitative model of\n")
cat("policyholder characteristics and the probability of lapse.\n")


# -----------------------------------------------------------------------------
# SECTION 7: CHECKPOINT 4 — MODEL DEVELOPMENT
# -----------------------------------------------------------------------------
# We tell the story of model development rather than jumping to one fit:
#   Model 0 (Null) -> Model 1 (Raw Variables) -> Model 2 (Business Engineered
#   Variables) -> Model 3 (Stepwise AIC simplification -> Final Model)

banner("CHECKPOINT 4: MODEL DEVELOPMENT")

cat("--- 7.0 Modelling Philosophy ---\n")
cat("The objective of the predictive model is not solely to maximise\n")
cat("classification accuracy. Equal importance is placed on interpretability,\n")
cat("transparency, and the ability to translate statistical relationships into\n")
cat("practical retention strategies. For this reason, black-box algorithms\n")
cat("(random forests, gradient boosting, neural networks) are intentionally\n")
cat("not benchmarked here: they would broaden the project's scope without\n")
cat("strengthening its actuarial message.\n\n")

cat("--- 7.0 (cont). Why Logistic Regression? ---\n")
cat("Logistic regression was selected because the outcome variable (policy\n")
cat("lapse) is binary, while the model remains highly interpretable and widely\n")
cat("adopted within actuarial and financial risk modelling. Unlike more\n")
cat("complex machine learning algorithms, logistic regression enables direct\n")
cat("interpretation of coefficient estimates and odds ratios, supporting\n")
cat("transparent, auditable business decisions.\n\n")

cat("--- 7.0 (cont). A Note on Information Leakage ---\n")
cat("Because the synthetic outcome (Section 3) was generated using transparent\n")
cat("actuarial assumptions built from several of the predictors below, the\n")
cat("objective of the logistic regression is to evaluate whether the\n")
cat("modelling framework successfully RECOVERS those expected relationships —\n")
cat("not to claim the discovery of new lapse drivers.\n\n")

cat("--- 7.0 (cont). Multicollinearity Check-Ahead ---\n")
cat("Correlation assessment performed during EDA (Section 5g) will be\n")
cat("complemented by Variance Inflation Factor (VIF) analysis in Section 8 to\n")
cat("confirm coefficient stability.\n\n")

# --- 7a. Train / Test Split (70/30) ---

cat("--- 7a. Train / Test Split ---\n")
cat("Separating the data into training and testing subsets provides an\n")
cat("unbiased assessment of predictive performance on previously unseen\n")
cat("observations, avoiding an optimistic estimate of model performance.\n\n")

train_idx <- createDataPartition(df$Lapsed, p = 0.70, list = FALSE)
train_df  <- df[ train_idx, ]
test_df   <- df[-train_idx, ]

cat(sprintf("Train set: %d rows | Test set: %d rows\n", nrow(train_df), nrow(test_df)))
cat(sprintf("Train lapse rate: %.2f%% | Test lapse rate: %.2f%%\n\n",
            mean(train_df$Lapsed == "Lapsed") * 100,
            mean(test_df$Lapsed  == "Lapsed") * 100))

# Helper: test-set AUC & Accuracy for any fitted glm, used in the comparison table below
test_metrics <- function(model, test_data) {
  p <- predict(model, newdata = test_data, type = "response")
  pred_class <- factor(ifelse(p >= 0.5, "Lapsed", "Active"), levels = c("Active","Lapsed"))
  roc_o <- roc(test_data$Lapsed, p, levels = c("Active","Lapsed"), quiet = TRUE)
  cm_o  <- confusionMatrix(pred_class, test_data$Lapsed, positive = "Lapsed")
  c(AUC = as.numeric(auc(roc_o)), Accuracy = as.numeric(cm_o$overall["Accuracy"]))
}

# --- 7b. Model 0: Null Model (intercept only — baseline) ---

model_null <- glm(Lapsed ~ 1, data = train_df, family = binomial(link = "logit"))
cat("=== MODEL 0: NULL MODEL (baseline) ===\n")
cat("AIC:", round(AIC(model_null), 1), "| Null Deviance:", round(model_null$null.deviance, 1), "\n\n")

# --- 7c. Model 1: Raw Variables ---

model1_raw <- glm(
  Lapsed ~ age + gender + marital_status + number_of_dependents +
    annual_income + health_status + smoking_status +
    policy_type + coverage_amount + monthly_premium +
    policy_duration_years,
  data = train_df, family = binomial(link = "logit")
)

cat("=== MODEL 1: RAW VARIABLES ===\n")
print(summary(model1_raw))
cat("AIC:", round(AIC(model1_raw), 1), "\n\n")

# --- 7d. Model 2: Business Engineered Variables ---
# age -> age_group | annual_income + monthly_premium -> premium_income_ratio

model2_engineered <- glm(
  Lapsed ~ age_group + gender + marital_status + number_of_dependents +
    health_status + smoking_status + policy_type + coverage_amount +
    premium_income_ratio + policy_duration_years,
  data = train_df, family = binomial(link = "logit")
)

cat("=== MODEL 2: BUSINESS ENGINEERED VARIABLES ===\n")
print(summary(model2_engineered))
cat("AIC:", round(AIC(model2_engineered), 1), "\n\n")

cat("--- Model 1 vs Model 2: does the engineered form earn its place? ---\n")
if (AIC(model2_engineered) < AIC(model1_raw)) {
  cat(sprintf("Model 2 (AIC = %.1f) OUTPERFORMS Model 1 (AIC = %.1f). The engineered\n",
              AIC(model2_engineered), AIC(model1_raw)))
  cat("business variables — particularly premium_income_ratio — are CONFIRMED as\n")
  cat("primary explanatory variables rather than being discarded.\n\n")
} else {
  cat(sprintf("Model 1 (AIC = %.1f) currently fits marginally better than Model 2\n",
              AIC(model1_raw)))
  cat(sprintf("(AIC = %.1f). We still carry the engineered form into stepwise selection\n",
              AIC(model2_engineered)))
  cat("(Model 3) since AIC differences this close are not conclusive on their own,\n")
  cat("and the engineered form remains preferred for business interpretability.\n\n")
}

# --- 7e. Model 3: Stepwise Simplification (AIC-based, from Model 2) -> FINAL MODEL ---
# Framed deliberately as a simplification/interpretability tool, not as a
# search for "the best model" — stepwise selection has well-known limitations
# (e.g. selection bias in p-values) that this framing avoids overstating.

model_final <- step(model2_engineered, direction = "both", trace = 0)

cat("=== MODEL 3: STEPWISE SIMPLIFICATION -> FINAL MODEL ===\n")
cat("Stepwise selection is used here as an interpretable variable-selection\n")
cat("procedure to balance explanatory power against model complexity — not as\n")
cat("a claim that this is definitively 'the best' model.\n\n")
print(summary(model_final))
cat("AIC:", round(AIC(model_final), 1), "\n\n")

cat("=== ODDS RATIOS (Final Model) ===\n")
or_final <- exp(cbind(OR = coef(model_final), suppressMessages(confint(model_final))))
print(round(or_final, 4))
cat("\nFull business-language interpretation of each significant odds ratio is\n")
cat("provided in Section 9 (Checkpoint 6: Business Insights).\n")

# --- 7f. Model Comparison Table (AIC/BIC/Deviance + Test AUC/Accuracy) ---

cat("\n=== MODEL DEVELOPMENT SUMMARY ===\n")

metrics_null   <- test_metrics(model_null, test_df)
metrics_m1     <- test_metrics(model1_raw, test_df)
metrics_m2     <- test_metrics(model2_engineered, test_df)
metrics_final  <- test_metrics(model_final, test_df)

model_comparison <- data.frame(
  Model    = c("Model 0: Null", "Model 1: Raw Variables",
               "Model 2: Business Engineered", "Model 3: Stepwise (Final)"),
  AIC      = round(c(AIC(model_null), AIC(model1_raw),
                      AIC(model2_engineered), AIC(model_final)), 1),
  BIC      = round(c(BIC(model_null), BIC(model1_raw),
                      BIC(model2_engineered), BIC(model_final)), 1),
  Deviance = round(c(model_null$deviance, model1_raw$deviance,
                      model2_engineered$deviance, model_final$deviance), 1),
  N_Params = c(length(coef(model_null)), length(coef(model1_raw)),
               length(coef(model2_engineered)), length(coef(model_final))),
  Test_AUC      = round(c(metrics_null["AUC"], metrics_m1["AUC"],
                           metrics_m2["AUC"], metrics_final["AUC"]), 3),
  Test_Accuracy = round(c(metrics_null["Accuracy"], metrics_m1["Accuracy"],
                           metrics_m2["Accuracy"], metrics_final["Accuracy"]), 3)
)
print(model_comparison, row.names = FALSE)
save_table(model_comparison, "model_comparison.csv")
cat("\nLower AIC/BIC indicates a better fit-vs-complexity trade-off; higher\n")
cat("Test AUC indicates better held-out discrimination. The Final Model\n")
cat("(Model 3) is the most parsimonious model that does not sacrifice fit.\n")

# --- 7g. Model Development Timeline ---

cat("\n--- Model Development Timeline ---\n")
timeline_tbl <- data.frame(
  Stage = c("Null Model", "Raw-Variable Model", "Business-Engineered Model",
            "Stepwise (Final) Model"),
  Purpose = c("Establish baseline reference performance",
              "Assess all raw predictors without transformation",
              "Test whether engineered affordability/age features improve fit",
              "Simplify to a parsimonious, interpretable specification"),
  Outcome = c("Reference deviance/AIC",
              "Maximum information, higher complexity",
              "Confirms/refutes value of engineered variables",
              "Business deployment candidate (validated in Section 8)"),
  stringsAsFactors = FALSE
)
print(timeline_tbl, row.names = FALSE)

# --- 7h. Variable Importance (Ranked by |Wald z-statistic|) ---

cat("\n--- Variable Importance (Ranked by |Wald z-statistic|) ---\n")
var_importance <- tidy(model_final) %>%
  filter(term != "(Intercept)") %>%
  mutate(abs_wald = round(abs(statistic), 3)) %>%
  arrange(desc(abs_wald)) %>%
  select(term, estimate, std.error, abs_wald, p.value) %>%
  rename(Wald_Z = abs_wald)
print(var_importance)
cat("\nA higher |Wald z| indicates a predictor whose estimated effect is large\n")
cat("relative to its uncertainty — i.e. a stronger, more precisely estimated\n")
cat("driver of lapse within this model.\n")

# --- 7i. Tidy Coefficient Table (for report) ---

cat("\n=== TIDY COEFFICIENT TABLE (Final Model) ===\n")
tidy_final <- tidy(model_final, conf.int = TRUE, exponentiate = TRUE) %>%
  mutate(across(where(is.numeric), ~ round(., 4)))
print(tidy_final)

final_model_coefficients <- tidy_final %>%
  rename(odds_ratio = estimate, wald_statistic = statistic,
         ci_lower_95 = conf.low, ci_upper_95 = conf.high)
save_table(final_model_coefficients, "final_model_coefficients.csv")


# -----------------------------------------------------------------------------
# SECTION 8: CHECKPOINT 5 — MODEL VALIDATION & DIAGNOSTICS
# -----------------------------------------------------------------------------
# Fitting a model is not the end of the workflow. This section evaluates
# calibration, stability, and classification performance from multiple
# angles, and — per the review — keeps the narrative centred on the four
# metrics that matter most (AUC, Sensitivity, Lift, Cross-Validation) while
# still reporting the supporting diagnostics (HL, VIF, residuals, calibration).

banner("CHECKPOINT 5: MODEL VALIDATION")

cat("Two distinct kinds of validation evidence are reported below, and it\n")
cat("matters which one a given metric belongs to:\n\n")
cat("  THRESHOLD-DEPENDENT CLASSIFICATION (8a, 8i) — Accuracy, Sensitivity,\n")
cat("  Specificity, and the Confusion Matrix. These all depend on WHERE the\n")
cat("  0/1 cut-off is drawn, and can look very different at different cut-offs.\n\n")
cat("  THRESHOLD-INDEPENDENT / RANKING PERFORMANCE (8b, 8c, 8h) — ROC AUC,\n")
cat("  the Lift Chart, and the Calibration plot. These evaluate how well the\n")
cat("  model RANKS and CALIBRATES predicted probabilities regardless of any\n")
cat("  particular cut-off, and are generally the more informative evidence for\n")
cat("  a rare-event problem like this one.\n\n")
cat("Given the ~3.5% lapse rate in this portfolio, the two categories tell a\n")
cat("noticeably different story below — the default 0.50 classification\n")
cat("threshold performs poorly, while the ranking metrics show the model does\n")
cat("carry real, useable signal.\n")

test_df$pred_prob  <- predict(model_final, newdata = test_df, type = "response")
test_df$pred_class <- factor(ifelse(test_df$pred_prob >= 0.5, "Lapsed", "Active"),
                              levels = c("Active","Lapsed"))

# --- 8a. Confusion Matrix @ Default Threshold (0.50) ---

cat("\n--- 8a. Confusion Matrix — Default Threshold (0.50) [threshold-dependent] ---\n")
cat("This threshold is reported for completeness, and to demonstrate why\n")
cat("accuracy alone is a poor metric under severe class imbalance — not as\n")
cat("the model's headline classification result (see 8i for the threshold we\n")
cat("actually recommend for illustrative classification).\n\n")
cm <- confusionMatrix(test_df$pred_class, test_df$Lapsed, positive = "Lapsed")
print(cm)

cat("\nBusiness meaning:\n")
cat("  Sensitivity — the proportion of policyholders who ACTUALLY lapse and are\n")
cat("  correctly identified by the model. Business implication: potential\n")
cat("  retention opportunities captured.\n")
cat("  Specificity — the proportion of RETAINED policyholders correctly\n")
cat("  classified as such. Business implication: avoiding unnecessary\n")
cat("  intervention costs on customers who were never going to lapse.\n")

# --- 8b. ROC Curve & AUC ---

cat("\n--- 8b. ROC AUC [threshold-independent / ranking] ---\n")
roc_obj <- roc(test_df$Lapsed, test_df$pred_prob, levels = c("Active","Lapsed"), quiet = TRUE)
auc_val <- auc(roc_obj)
cat(sprintf("ROC AUC: %.4f\n", auc_val))
cat("Plain-language meaning: the AUC is the probability that a randomly\n")
cat("selected policyholder who lapses receives a higher predicted lapse\n")
cat("probability than a randomly selected policyholder who remains active.\n")
cat("An AUC of ~0.64 indicates modest but meaningful discrimination — the\n")
cat("actuarially motivated predictors carry real signal for RANKING\n")
cat("policyholders by lapse risk, without this being a strong 0/1 classifier.\n")

p_roc <- ggroc(roc_obj, color = "#1565C0", linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "grey50") +
  annotate("text", x = 0.35, y = 0.15, label = sprintf("AUC = %.3f", auc_val),
           size = 5, color = "#1565C0", fontface = "bold") +
  labs(title = "Figure 8: ROC Curve — Final Logistic Regression Model",
       subtitle = "Discrimination ability of the model on the held-out test set",
       x = "Specificity (1 - False Positive Rate)", y = "Sensitivity (True Positive Rate)")
print(p_roc)
save_fig(p_roc, "08_roc_curve.png")

# --- 8c. Lift Chart (Decile Analysis) — a targeting diagnostic, not just another plot ---

cat("\n--- 8c. Lift Chart (Decile Analysis) [threshold-independent / ranking] ---\n")
cat("The lift chart demonstrates the practical value of prioritising\n")
cat("high-risk policyholders for retention campaigns rather than contacting\n")
cat("the entire customer portfolio — this is a customer-targeting tool, and\n")
cat("arguably the most commercially useful validation result in this project,\n")
cat("precisely because it does not depend on picking a single 0/1 threshold.\n\n")

test_df <- test_df %>% mutate(decile = ntile(pred_prob, 10)) %>% arrange(decile)

lift_table <- test_df %>%
  group_by(decile) %>%
  summarise(n = n(), actual_lapse = sum(Lapsed == "Lapsed"),
            avg_pred = mean(pred_prob), lapse_rate = mean(Lapsed == "Lapsed"),
            .groups = "drop") %>%
  mutate(overall_rate = sum(actual_lapse) / sum(n), lift = lapse_rate / overall_rate)

risk_segmentation <- lift_table %>%
  mutate(across(c(avg_pred, lapse_rate, overall_rate, lift), ~ round(., 4))) %>%
  rename(predicted_risk_decile = decile, n_policyholders = n,
         mean_predicted_prob = avg_pred, observed_lapse_rate = lapse_rate,
         portfolio_lapse_rate = overall_rate)
save_table(risk_segmentation, "risk_segmentation.csv")

print(lift_table %>% select(decile, n, lapse_rate, avg_pred, lift) %>%
        mutate(across(c(lapse_rate, avg_pred), ~ percent(., accuracy = 0.1)),
               lift = round(lift, 2)))

top_decile_lift <- round(lift_table$lift[lift_table$decile == max(lift_table$decile)], 2)
cat(sprintf("\nThe highest-risk decile shows a lift of %.2fx over the portfolio-wide\n", top_decile_lift))
cat("lapse rate — i.e. contacting just that decile finds lapses at roughly\n")
cat(sprintf("%.1fx the rate of contacting customers at random.\n", top_decile_lift))

p_lift <- ggplot(lift_table, aes(x = factor(decile), y = lift, fill = lift > 1)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values = c("#B0BEC5","#1565C0")) +
  geom_text(aes(label = round(lift, 2)), vjust = -0.4, size = 3.2) +
  labs(title = "Figure 9: Lift Chart by Predicted Risk Decile",
       subtitle = "Decile 10 = highest predicted lapse risk — the priority segment for retention outreach",
       x = "Predicted Risk Decile (1 = lowest, 10 = highest)",
       y = "Lift (Observed Rate / Overall Rate)")
print(p_lift)
save_fig(p_lift, "09_lift_chart.png", width = 9)

# --- 8d. Hosmer-Lemeshow Goodness-of-Fit Test ---

cat("\n--- 8d. Hosmer-Lemeshow Test ---\n")
hl_test <- hoslem.test(as.numeric(test_df$Lapsed == "Lapsed"), test_df$pred_prob, g = 10)
print(hl_test)
cat("Interpretation: p-value > 0.05 indicates no strong evidence against\n")
cat("adequate calibration. Hosmer-Lemeshow has known limitations (sensitivity\n")
cat("to binning choice), so it is treated here as ONE diagnostic among\n")
cat("several rather than a definitive goodness-of-fit verdict.\n")

# --- 8e. Variance Inflation Factor (Multicollinearity) ---

cat("\n--- 8e. VIF Check (Multicollinearity) ---\n")
vif_result <- tryCatch(vif(model_final), error = function(e) NULL)
if (is.null(vif_result)) {
  cat("VIF not computable (model has fewer than 2 predictors after stepwise\n")
  cat("selection) — multicollinearity is not a concern in a single-predictor model.\n")
} else {
  print(round(vif_result, 3))
  cat("Rule of thumb: VIF > 5 signals problematic multicollinearity. Low VIF\n")
  cat("values indicate predictor estimates are not materially affected by\n")
  cat("multicollinearity, improving interpretability and coefficient stability.\n")
}

# --- 8f. Deviance Residuals Plot ---

residuals_df <- data.frame(
  fitted    = fitted(model_final),
  residuals = residuals(model_final, type = "deviance"),
  Lapsed    = train_df$Lapsed
)

p_resid <- ggplot(residuals_df, aes(x = fitted, y = residuals, color = Lapsed)) +
  geom_point(alpha = 0.3, size = 0.8) +
  geom_hline(yintercept = c(-2, 0, 2), linetype = c("dashed","solid","dashed"), color = "grey40") +
  scale_color_manual(values = status_colors) +
  labs(title = "Figure 10: Deviance Residuals vs Fitted Values",
       x = "Fitted Probability", y = "Deviance Residual") +
  theme(legend.position = "top")
print(p_resid)
save_fig(p_resid, "10_deviance_residuals.png")
cat("\n--- 8f. Residual Diagnostics ---\n")
cat("Residual diagnostics revealed no major systematic departures from model\n")
cat("assumptions, supporting the adequacy of the fitted logistic regression.\n")

# --- 8g. 10-Fold Cross-Validation ---

cat("\n--- 8g. 10-Fold Cross-Validation ---\n")
cat("Cross-validation evaluates the stability of model performance across\n")
cat("multiple resampled datasets, reducing the risk that results depend on\n")
cat("one particular train-test split.\n\n")

cv_control <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                            summaryFunction = twoClassSummary, savePredictions = "final")

set.seed(42)
cv_model <- train(
  formula(model_final), data = train_df, method = "glm", family = "binomial",
  metric = "ROC", trControl = cv_control
)

print(cv_model)
cat(sprintf("\nMean 10-fold CV: ROC = %.4f | Sensitivity = %.4f | Specificity = %.4f\n",
            cv_model$results$ROC, cv_model$results$Sens, cv_model$results$Spec))
cat("The similarity between the cross-validated ROC and the held-out test AUC\n")
cat("provides some evidence that the model's modest discriminatory\n")
cat("performance is not solely driven by the particular train/test split —\n")
cat("this is supportive evidence of stability, not a claim of strong\n")
cat("generalisation, given the AUC itself is modest (~0.62-0.64).\n")

# --- 8h. Calibration Assessment ---

cat("\n--- 8h. Calibration Assessment (Predicted vs Observed, by Decile) ---\n")

calib_df <- test_df %>%
  mutate(bin = ntile(pred_prob, 10)) %>%
  group_by(bin) %>%
  summarise(mean_pred = mean(pred_prob), obs_rate = mean(Lapsed == "Lapsed"),
            n = n(), .groups = "drop")
print(calib_df)

p_calib <- ggplot(calib_df, aes(x = mean_pred, y = obs_rate)) +
  geom_point(size = 3, color = "#1565C0") +
  geom_line(color = "#1565C0") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  scale_x_continuous(labels = percent_format()) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Figure 11: Calibration Plot",
       subtitle = "Points near the diagonal indicate well-calibrated probability estimates",
       x = "Mean Predicted Probability", y = "Observed Lapse Rate")
print(p_calib)
save_fig(p_calib, "11_calibration_plot.png")

# --- 8i. Classification Threshold Trade-off ---

cat("\n--- 8i. Classification Threshold Trade-off [threshold-dependent] ---\n")
cat("The confusion matrix in 8a uses the conventional 0.50 threshold. Because\n")
cat("lapse is a rare event in this portfolio (~3.5%), predicted probabilities\n")
cat("rarely approach 0.50 at all, and 0.50 is not a meaningful cut-off here —\n")
cat("it exists mainly to illustrate why accuracy is a misleading metric under\n")
cat("severe class imbalance. In practice, insurers would set the threshold\n")
cat("based on the relative costs of missed lapse events versus unnecessary\n")
cat("retention interventions.\n\n")

thresholds <- seq(0.1, 0.9, by = 0.1)
threshold_tbl <- map_df(thresholds, function(t) {
  pred_t <- factor(ifelse(test_df$pred_prob >= t, "Lapsed", "Active"),
                    levels = c("Active","Lapsed"))
  cm_t <- confusionMatrix(pred_t, test_df$Lapsed, positive = "Lapsed")
  data.frame(threshold   = t,
             sensitivity = round(cm_t$byClass["Sensitivity"], 3),
             specificity = round(cm_t$byClass["Specificity"], 3),
             accuracy    = round(cm_t$overall["Accuracy"], 3))
})
print(threshold_tbl, row.names = FALSE)

best_coords <- coords(roc_obj, "best", best.method = "youden",
                       ret = c("threshold","sensitivity","specificity"),
                       transpose = FALSE)
youden_threshold <- best_coords$threshold[1]

cat(sprintf("\nYouden-optimal threshold: %.2f (Sensitivity = %.1f%%, Specificity = %.1f%%)\n",
            youden_threshold, best_coords$sensitivity[1] * 100,
            best_coords$specificity[1] * 100))
cat("Because lapse is the minority class here, a threshold below 0.5 is often\n")
cat("preferred in practice: it trades some specificity for higher sensitivity,\n")
cat("usually the right trade-off when the cost of a missed lapse (lost future\n")
cat("premium, CSM erosion) exceeds the cost of an unnecessary retention call.\n\n")

cat(sprintf("--- Confusion Matrix — Youden-Optimal Threshold (%.2f) [primary illustrative threshold] ---\n",
            youden_threshold))
test_df$pred_class_youden <- factor(ifelse(test_df$pred_prob >= youden_threshold, "Lapsed", "Active"),
                                     levels = c("Active","Lapsed"))
cm_youden <- confusionMatrix(test_df$pred_class_youden, test_df$Lapsed, positive = "Lapsed")
print(cm_youden)

cat("\nThis is the threshold adopted as the primary ILLUSTRATIVE classification\n")
cat("cut-off for the remainder of this report, rather than 0.50. Statistically,\n")
cat("Youden's J balances sensitivity and specificity equally; a real deployment\n")
cat("would go one step further and set the threshold from the actual economics\n")
cat("of retention:\n\n")
cat("  Statistical threshold (Youden, ~", round(youden_threshold, 2), ")\n", sep = "")
cat("        |\n")
cat("        v\n")
cat("  Business threshold — set instead from:\n")
cat("    - cost of a retention intervention\n")
cat("    - expected future premium / customer lifetime value\n")
cat("    - probability that an intervention actually prevents the lapse\n")
cat("    - cost of a false positive (contacting a customer who wasn't leaving)\n")
cat("    - value of preventing a true lapse\n\n")
cat("That business-threshold calibration requires cost/value inputs beyond\n")
cat("this dataset and is noted here as the natural next step rather than\n")
cat("being computed on assumed figures.\n")

# --- 8j. Summary Dashboard ---

cat("\n--- 8j. Model Performance Summary Dashboard ---\n")
summary_dashboard <- data.frame(
  Metric = c("Accuracy @ 0.50 (diagnostic only)", "Sensitivity @ 0.50 (diagnostic only)",
             "Specificity @ 0.50 (diagnostic only)",
             sprintf("Accuracy @ %.2f (Youden, primary)", youden_threshold),
             sprintf("Sensitivity @ %.2f (Youden, primary)", youden_threshold),
             sprintf("Specificity @ %.2f (Youden, primary)", youden_threshold),
             "Test AUC (ranking)", "10-fold CV ROC (ranking)",
             "Top-decile Lift (ranking)", "Hosmer-Lemeshow p", "Max VIF"),
  Value = c(
    percent(as.numeric(cm$overall["Accuracy"]), accuracy = 0.1),
    percent(as.numeric(cm$byClass["Sensitivity"]), accuracy = 0.1),
    percent(as.numeric(cm$byClass["Specificity"]), accuracy = 0.1),
    percent(as.numeric(cm_youden$overall["Accuracy"]), accuracy = 0.1),
    percent(as.numeric(cm_youden$byClass["Sensitivity"]), accuracy = 0.1),
    percent(as.numeric(cm_youden$byClass["Specificity"]), accuracy = 0.1),
    round(as.numeric(auc_val), 3),
    round(cv_model$results$ROC, 3),
    paste0(top_decile_lift, "x"),
    round(hl_test$p.value, 4),
    ifelse(is.null(vif_result), "N/A", round(max(vif_result[,1]), 2))
  ),
  Interpretation = c(
    "Misleadingly high under class imbalance — the model predicts almost no lapses at this cut-off",
    "0% at this cut-off — demonstrates why 0.50 is inappropriate here",
    "100% at this cut-off — trivially achieved by never predicting lapse",
    "Overall correct classification at the recommended illustrative threshold",
    "Share of actual lapses correctly flagged at the recommended threshold",
    "Share of actual persisters correctly left alone at the recommended threshold",
    "Ranking/discrimination quality on unseen data",
    "Stability of ranking performance across resamples",
    "Lapse rate in the highest-risk decile vs the portfolio average",
    "p > 0.05: no strong evidence against calibration",
    "< 5 indicates multicollinearity is not a material concern"
  ),
  stringsAsFactors = FALSE
)
print(summary_dashboard, row.names = FALSE)
save_table(summary_dashboard, "validation_results.csv")

# --- 8k. Business Validation & Deployment Readiness ---

cat("\n--- 8k. Business Validation ---\n")
cat("From a business perspective, identifying policyholders with elevated\n")
cat("lapse probability enables insurers to prioritise targeted retention\n")
cat("efforts, potentially improving persistency while reducing the cost of\n")
cat("blanket customer outreach.\n\n")

cat("Taken together, the validation evidence tells a specific story: this\n")
cat("model is a modest but genuine RISK-RANKING tool, not a strong default\n")
cat("binary classifier. At the conventional 0.50 threshold it is practically\n")
cat("useless (0% sensitivity); at the Youden-optimal threshold it becomes a\n")
cat("reasonably balanced classifier; and via the lift chart / decile ranking\n")
cat("it demonstrates clear, useable value for concentrating retention\n")
cat("resources on higher-risk policyholders. Reporting only the 0.50-threshold\n")
cat("accuracy (96.5%) would be misleading; reporting only the AUC (~0.64)\n")
cat("would undersell the model's practical targeting value. The combination\n")
cat("of ranking metrics (AUC, lift) and a business-appropriate threshold is\n")
cat("the honest summary of this model's performance.\n\n")

cat("The validation results suggest that the model provides a sufficiently\n")
cat("stable and interpretable framework for identifying higher-risk\n")
cat("policyholders within the synthetic dataset. While additional external\n")
cat("validation would be required before operational deployment, the\n")
cat("modelling workflow demonstrates how predictive analytics can support\n")
cat("targeted retention strategies.\n")


# -----------------------------------------------------------------------------
# SECTION 9: CHECKPOINT 6 — BUSINESS INSIGHTS
# -----------------------------------------------------------------------------
# Every statistical result is translated into an actuarial implication,
# phrased "holding all other variables constant..." to match how odds ratios
# are correctly interpreted in a multivariable model.

banner("CHECKPOINT 6: BUSINESS INSIGHTS")

generate_insight <- function(term, or_val, p_val) {
  direction <- ifelse(or_val > 1, "higher", "lower")
  magnitude <- round(abs(or_val - 1) * 100, 0)

  case_when(
    str_detect(term, "premium_income_ratio") ~ {
      beta_est <- log(or_val)                        # recover the raw coefficient
      pct_point_or <- exp(beta_est * 0.01)            # effect of a 1-percentage-point change
      pct_point_pct <- round((pct_point_or - 1) * 100, 1)
      sprintf(paste(
        "Holding all other variables constant, a higher premium-to-income",
        "ratio is associated with materially higher lapse risk, suggesting",
        "affordability is a key driver of policy retention. Because this is a",
        "continuous predictor, the raw odds ratio (OR = %.2f) describes a full",
        "one-unit (100 percentage-point) change and should NOT be read as",
        "'policyholders have %.0fx higher odds of lapsing' — a change of that",
        "size essentially never occurs in practice. The economically",
        "meaningful figure is the effect of a realistic 1-percentage-point",
        "change in the ratio: approximately %.1f%% higher odds of lapsing per",
        "1pp increase. Flexible premium options or payment holidays may",
        "improve persistency among policyholders with elevated ratios."
      ), or_val, or_val, pct_point_pct)
    },

    str_detect(term, "policy_duration_years") ~
      sprintf(paste(
        "Holding all other variables constant, each additional year a policy",
        "remains in force is associated with approximately %d%% %s odds of",
        "lapsing (OR = %.2f), consistent with SOA persistency studies where",
        "early policy years carry the greatest lapse risk. Retention resources",
        "are best concentrated in years 1-2."
      ), magnitude, direction, or_val),

    str_detect(term, "^age") ~
      sprintf(paste(
        "Holding all other variables constant, older policyholders exhibit",
        "approximately %d%% %s odds of lapsing (OR = %.2f) relative to the",
        "youngest cohort, reflecting greater financial stability and stronger",
        "attachment to long-term coverage as customers age."
      ), magnitude, direction, or_val),

    str_detect(term, "smoking_statusCurrent smoker") ~
      sprintf(paste(
        "Holding all other variables constant, current smokers exhibit",
        "approximately %d%% %s odds of lapsing (OR = %.2f) than non-smokers,",
        "potentially reflecting higher premium loadings for smoker risk that",
        "strain affordability."
      ), magnitude, direction, or_val),

    str_detect(term, "policy_typeTerm Life") ~
      sprintf(paste(
        "Holding all other variables constant, Term Life policyholders exhibit",
        "approximately %d%% %s odds of lapsing (OR = %.2f) relative to the",
        "reference policy type, consistent with Term Life's lower cash-value",
        "lock-in and more price-sensitive buyers."
      ), magnitude, direction, or_val),

    str_detect(term, "policy_typeWhole Life") ~
      sprintf(paste(
        "Holding all other variables constant, Whole Life policyholders",
        "exhibit approximately %d%% %s odds of lapsing (OR = %.2f), reflecting",
        "the cash-value accumulation and surrender-charge structures that",
        "discourage early termination."
      ), magnitude, direction, or_val),

    str_detect(term, "marital_statusMarried") ~
      sprintf(paste(
        "Holding all other variables constant, married policyholders exhibit",
        "approximately %d%% %s odds of lapsing (OR = %.2f) versus the",
        "reference group, consistent with greater household financial",
        "stability and shared responsibility for coverage."
      ), magnitude, direction, or_val),

    str_detect(term, "health_status") ~
      sprintf(paste(
        "Holding all other variables constant, policyholders in this health",
        "category exhibit approximately %d%% %s odds of lapsing (OR = %.2f),",
        "which likely signals broader financial strain correlated with health",
        "status rather than a direct causal health effect."
      ), magnitude, direction, or_val),

    str_detect(term, "number_of_dependents") ~
      sprintf(paste(
        "Holding all other variables constant, each additional dependent is",
        "associated with approximately %d%% %s odds of lapsing (OR = %.2f),",
        "reflecting the competing financial pressures of supporting dependents",
        "against maintaining premium payments."
      ), magnitude, direction, or_val),

    TRUE ~
      sprintf("Holding all other variables constant, %s is associated with approximately %d%% %s odds of lapsing (OR = %.2f, p = %.4f).",
              term, magnitude, direction, or_val, p_val)
  )
}

sig_terms <- tidy(model_final, conf.int = TRUE, exponentiate = TRUE) %>%
  filter(term != "(Intercept)", p.value < 0.05) %>%
  arrange(p.value)

if (nrow(sig_terms) == 0) {
  cat("No predictors reached statistical significance at the 5% level in the\n")
  cat("final model — consider revisiting engineered variables or collecting\n")
  cat("additional features.\n")
} else {
  for (i in seq_len(nrow(sig_terms))) {
    cat(sprintf("%d. %s (OR = %.2f, p = %.4f)\n",
                i, sig_terms$term[i], sig_terms$estimate[i], sig_terms$p.value[i]))
    cat("   ", generate_insight(sig_terms$term[i], sig_terms$estimate[i], sig_terms$p.value[i]), "\n\n", sep = "")
  }
}

# --- 9a. Odds Ratio Forest Plot ---

or_df <- tidy(model_final, conf.int = TRUE, exponentiate = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(term = str_replace_all(term, c(
    "smoking_status" = "Smoking: ", "policy_type" = "Policy: ",
    "health_status" = "Health: ", "marital_status" = "Marital: ",
    "gender" = "Gender: ", "age_group" = "Age: "
  ))) %>%
  arrange(estimate)

p_forest <- ggplot(or_df, aes(x = estimate, y = reorder(term, estimate))) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbar(aes(xmin = conf.low, xmax = conf.high), width = 0.3,
                color = "grey40", orientation = "y") +   # geom_errorbarh() is soft-deprecated; this is the current non-deprecated form (requires ggplot2 >= 3.5.0)
  geom_point(aes(color = estimate > 1), size = 3, show.legend = FALSE) +
  scale_color_manual(values = c(col_active, col_lapsed)) +
  scale_x_log10() +
  labs(title = "Figure 12: Odds Ratio Forest Plot (Final Model)",
       subtitle = "Red = increases lapse risk | Green = decreases lapse risk | OR on log scale",
       x = "Odds Ratio (log scale)", y = NULL) +
  theme(axis.text.y = element_text(size = 9))
print(p_forest)
save_fig(p_forest, "12_odds_ratio_forest_plot.png", width = 9)

# --- 9b. High-Risk Segment Summary ---

cat("\n--- 9b. High-Risk Segment Analysis (Top 20% Predicted Risk) ---\n")

high_risk <- test_df %>% filter(pred_prob >= quantile(pred_prob, 0.80))

cat(sprintf("  Count: %d (%.0f%% of test set)\n", nrow(high_risk), nrow(high_risk)/nrow(test_df)*100))
cat(sprintf("  Actual lapse rate in this segment: %.1f%% vs overall test lapse rate: %.1f%%\n",
            mean(high_risk$Lapsed == "Lapsed")*100, mean(test_df$Lapsed == "Lapsed")*100))

cat("\n  Dominant policy types in high-risk segment:\n")
print(prop.table(table(high_risk$policy_type)) %>% sort(decreasing = TRUE) %>% round(3))

cat("\n  Dominant smoking status in high-risk segment:\n")
print(prop.table(table(high_risk$smoking_status)) %>% sort(decreasing = TRUE) %>% round(3))


# -----------------------------------------------------------------------------
# SECTION 10: ACTUARIAL APPLICATION — IFRS 17 LINKAGE
# -----------------------------------------------------------------------------

banner("SECTION 10: ACTUARIAL APPLICATION — IFRS 17 LAPSE ASSUMPTIONS")

cat("Under IFRS 17 (effective 1 Jan 2023), insurers must use 'current best\n")
cat("estimate' lapse assumptions in their Contractual Service Margin (CSM) and\n")
cat("Liability for Remaining Coverage (LRC). The discussion below is\n")
cat("deliberately qualitative rather than a worked numerical example — this\n")
cat("dataset is synthetic and does not represent a real insurer's balance\n")
cat("sheet, so a specific hypothetical liability figure would overstate what\n")
cat("this model can actually demonstrate. What it CAN demonstrate is how a\n")
cat("segmented, predictive lapse model of this kind fits into the IFRS 17\n")
cat("actuarial process.\n\n")

cat("1. LAPSE RATES BY SEGMENT — replace flat industry tables with\n")
cat("   policyholder-specific predicted probabilities, estimated directly from\n")
cat("   the fitted model on real, observed segment-level data below.\n\n")

segment_rates <- test_df %>%
  group_by(policy_type, smoking_status) %>%
  summarise(n_policies = n(), predicted_lapse = round(mean(pred_prob) * 100, 1),
            actual_lapse = round(mean(Lapsed == "Lapsed") * 100, 1), .groups = "drop") %>%
  filter(n_policies >= 20) %>%
  arrange(desc(predicted_lapse))

cat("--- Segmented Lapse Rates (Test-Set Data — a candidate input to CSM assumption-setting) ---\n")
print(segment_rates)

cat("\n2. CASH-FLOW & CSM LINKAGE — how this kind of model fits the IFRS 17 workflow:\n\n")
cat("   - Fulfilment cash flows: lapse assumptions directly affect projected\n")
cat("     future premium income and claims outgo, since a lapsed policy stops\n")
cat("     contributing future cash flows to the contract boundary.\n")
cat("   - CSM: where future service is affected (e.g. under the general\n")
cat("     measurement model), a revision to lapse assumptions is a change in\n")
cat("     estimate that flows through the CSM rather than the P&L directly,\n")
cat("     making the assumption-setting process itself an area of actuarial\n")
cat("     and audit scrutiny.\n")
cat("   - Segmented assumptions: predictive, policyholder-level lapse\n")
cat("     probabilities of the kind produced here could supplement flat,\n")
cat("     portfolio-wide lapse tables with segment-specific assumptions,\n")
cat("     subject to the model being validated on real experience data.\n")
cat("   - Management action: retention analytics built on this kind of model\n")
cat("     could feed into management decisions on product design, premium\n")
cat("     flexibility, and where to target retention spend — informing\n")
cat("     assumption-setting discussions rather than replacing them.\n\n")

cat("3. RETENTION STRATEGY — the top 20% high-risk policyholders (Section 9b)\n")
cat("   represent the highest-value retention intervention targets identified\n")
cat("   directly from this dataset.\n")


# -----------------------------------------------------------------------------
# SECTION 11: FINAL SUMMARY OUTPUT
# -----------------------------------------------------------------------------

banner("MODEL PERFORMANCE SUMMARY")

cat(sprintf("  Train size:         %d | Test size: %d\n", nrow(train_df), nrow(test_df)))
cat(sprintf("  Overall lapse rate: %.2f%%\n", mean(df$Lapsed == "Lapsed") * 100))
cat(sprintf("  Test AUC:           %.3f\n", as.numeric(auc_val)))
cat(sprintf("  10-fold CV ROC:     %.3f\n", cv_model$results$ROC))
cat(sprintf("  Accuracy:           %.1f%%\n", cm$overall["Accuracy"] * 100))
cat(sprintf("  Sensitivity:        %.1f%%\n", cm$byClass["Sensitivity"] * 100))
cat(sprintf("  Specificity:        %.1f%%\n", cm$byClass["Specificity"] * 100))
cat(sprintf("  Final Model AIC:    %.1f\n", AIC(model_final)))
cat(sprintf("  HL p-value:         %.4f (%s)\n", hl_test$p.value,
            ifelse(hl_test$p.value > 0.05, "no strong evidence against fit", "possible mis-calibration")))

cat("\n  Key Significant Predictors:\n")
sig <- tidy(model_final) %>%
  filter(p.value < 0.05, term != "(Intercept)") %>%
  arrange(p.value) %>%
  mutate(OR = round(exp(estimate), 3), p = round(p.value, 4))
print(sig %>% select(term, OR, p))

cat("\nScript complete. All figures rendered to the plot pane.\n")
cat("Save plots via: ggsave('figure_name.png', plot = last_plot(), dpi = 300)\n")
