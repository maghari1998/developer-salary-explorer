# Developer Salary Explorer

**Interactive salary-bucket prediction and technology recommendations**  
*MIE1624 — Introduction to Data Science, Analytics and AI · Assignment 2 · Fall 2026*

### [Launch the live dashboard](https://developer-salary-explorer-du4z3zanvuw4wzd3eeipac.streamlit.app/)

This Streamlit application explores salary patterns in the **2026 Stack Overflow Annual Developer Survey**. It uses a custom **cumulative-threshold ordinal logistic regression** model to estimate a respondent's probability across **15 ordered annual salary buckets (USD)** and to rank hypothetical additions of reported technologies by their model-predicted effect on expected salary-bucket rank.

> **Educational use only.** Results represent predictive associations in survey responses, not individual salary quotes, labor-market forecasts, or evidence that learning a technology will cause higher pay.

## Try it online

**[Open Developer Salary Explorer](https://developer-salary-explorer-du4z3zanvuw4wzd3eeipac.streamlit.app/)**

1. **Enter your original survey answers:** country, years coding, years of professional experience, number of tools, role, role functions, contributor/manager status, organization size, education, and technologies used. Leave an answer blank if it is unknown.
2. **Explore the prediction:** inspect the full 15-bucket probability chart, most likely salary bracket, probability of that bracket, and expected *bucket rank* (not salary in dollars).
3. **Explore recommendations:** review up to 15 ranked candidate technologies and the top three under the current model criterion. Change the minimum training-support count to exclude rarely reported technologies.

No account or raw dataset upload is required by dashboard visitors.

## Model at a glance

| Component | Approach |
| --- | --- |
| Dataset | Provided `processed_data_assignment2_2026.csv`, derived from the 2026 Stack Overflow survey |
| Target | `ConvertedCompYearlyBucket`: 15 ordered salary intervals in USD |
| Population | Employed / self-employed survey respondents retained after a stratified train/test split |
| Input groups | 12 selected original survey question groups, encoded into 263 model columns |
| Preprocessing | Training-derived imputation, category vocabularies and numerical scaling |
| Classifier | 14 `sklearn.linear_model.LogisticRegression` cumulative-threshold models for 15 ordered classes |
| Final tuning | L1 regularization, `C = 0.3` |
| Probabilities | Corrected to nondecreasing cumulative probabilities before converting to class probabilities |
| Technology ranking | Change in expected ordered salary-bucket rank after adding one reported technology |

**Recorded held-out evaluation** in the accompanying assignment analysis: accuracy **30.74%**, ordinal MAE **1.5050 buckets**, and quadratic-weighted kappa **0.6871**. Exact salary-bucket accuracy is inherently difficult, and results are less dependable in sparse salary categories.

The model bundle used by this site is **pretrained**. The Streamlit app does not refit the model at each visit.

## Repository files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit web interface and charts |
| `assignment2_bonus_dashboard.py` | Shared original-answer encoding and technology-ranking functions |
| `model_bundle.joblib` | Fitted model, training-derived preprocessing mappings, and dropdown metadata |
| `requirements.txt` | Python package dependencies |
| `build_model_local.py` | Optional local reproduction of the deployment bundle from the official dataset |
| `README.md` | Project documentation |
| `.gitignore` | Excludes raw CSV files and local build clutter |

The deployed repository does **not** require the raw 71 MB survey CSV. The model bundle contains fitted model objects and derived input choices, not the individual survey records.

## Run locally

Use **Python 3.12**, which is compatible with this project's pinned packages.

```bash
# In the cloned/downloaded repository folder:
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux; on Windows use .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then visit the local URL shown by Streamlit (typically `http://localhost:8501`). Keep `model_bundle.joblib` alongside `app.py`.

## Deploy to Streamlit Community Cloud

1. Push all deployment files to the **root** of a GitHub repository. The raw survey CSV is **not** needed.
2. In [Streamlit Community Cloud](https://share.streamlit.io), choose **Create app** and select the GitHub repository.
3. Choose branch `main`, main file path **`app.py`**, and **Python 3.12** in advanced settings.
4. Deploy and check that all survey inputs, prediction results, and recommendation tables appear correctly.

Live deployment: **[developer-salary-explorer-du4z3zanvuw4wzd3eeipac.streamlit.app](https://developer-salary-explorer-du4z3zanvuw4wzd3eeipac.streamlit.app/)**

## Interpretation and limitations

- **Association is not causation.** The recommender changes survey-response indicators; it cannot estimate the causal payoff from learning a skill.
- **Support matters.** A high-ranked technology may have few reporting respondents and unstable associations. The dashboard offers a minimum training-support filter; support is not the same as confidence.
- **Unequal bucket widths.** An improvement of one expected bucket rank does not represent a fixed number of dollars.
- **Sampling and generalization.** Survey participation is self-selected, and the analyzed group excludes people outside the chosen employment categories. Unusual profiles and tail salary brackets are especially uncertain.
- **Model validation.** Feature selection and some preprocessing were performed on the entire training partition before cross-validation, so CV estimates may be optimistic. The held-out test data was reserved for final evaluation.
- **Probability quality.** The model repairs crossings between independently trained cumulative thresholds, but valid probability distributions are not necessarily well-calibrated.
- **Model security.** `joblib` loads Python objects. Use only the trusted bundle included with this repository.

## Academic context

Created for **MIE1624 Assignment 2 (Fall 2026), University of Toronto**. The dashboard implements the optional interactive extension of the Q5 individual prediction and technology-recommender workflow. For detailed methodology, experiments, comparisons, limitations, and AI-use disclosure, refer to the accompanying assignment report and Jupyter notebook.
