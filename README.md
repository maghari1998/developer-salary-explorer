# Developer Salary Explorer — MIE1624 Assignment 2

A Streamlit dashboard for the 2026 Stack Overflow Annual Developer Survey, based on a custom `K-1` cumulative-threshold ordinal logistic regression. The app uses the original question groups, displays predicted probabilities for 15 salary buckets, and ranks hypothetical technology indicators by predicted expected-bucket-rank change. These are **associations, not causal salary effects**.

## Deploy using Streamlit Community Cloud

1. Unzip the package and upload the individual files to the **root of a GitHub repository**, including `app.py`, `assignment2_bonus_dashboard.py`, `model_bundle.joblib`, and `requirements.txt`.
2. Open <https://share.streamlit.io> and select **Create app** > **Yup, I have an app**.
3. Choose your GitHub repository, branch `main`, and entrypoint **`app.py`**.
4. Select **Deploy**. Choose a `.streamlit.app` subdomain if desired. Once deployed, share the URL.

Do **not** upload the original 71 MB survey CSV: the hosted app already contains the trained model and aggregated dropdown choices. The bundle omits respondent-level records. It was generated from the unmodified dataset and can be reproduced locally with `build_model_local.py`, but this is not required for deployment.

## Local launch

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Limitations

The feature-selection stage was performed using the full training partition before cross-validation (as described in the assignment report), so CV results may be optimistic. A salary-bucket increment is not a fixed dollar gain. Rare technologies may have unstable associations. The saved fitted model should only be loaded from a trusted source.
