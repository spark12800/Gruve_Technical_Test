# Gruve AI Engineering Assignment — Diabetes Hospital Readmission Prediction

## Project Overview

This project predicts whether a diabetic patient will be readmitted to hospital within 30 days (`<30`), after 30 days (`>30`), or not at all (`NO`), using the UCI Diabetes 130-US Hospitals dataset.

The training set contains **77,439 patient encounters × 46 features**, with significant class imbalance: NO 53.8%, >30 35.0%, <30 11.2% (imbalance ratio 4.8×). The pipeline uses **AutoGluon TabularPredictor** with LightGBM, CatBoost, and a Weighted Ensemble, combined with SMOTENC oversampling, domain-specific feature engineering (Charlson Comorbidity Index), SHAP explainability, and threshold tuning for the minority class.

---

## Repository Structure

```
.
├── Gruve_SieunPark.ipynb     # Main notebook — full pipeline (EDA → training → evaluation)
├── diabetic_data.csv         # Raw dataset (UCI Diabetes 130-US Hospitals)
├── IDS_mapping.csv           # ID-to-label mappings for categorical columns
├── leaderboard.csv           # AutoGluon model leaderboard (validation scores)
├── unseen_data.csv           # 5% held-out patient sample for grader inferencing
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Environment Setup

This project uses `uv` for environment management.

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows

uv pip install -r requirements.txt
```

---

## Running the Training Script

Open and run all cells in the notebook:

```bash
jupyter notebook Gruve_SieunPark.ipynb
```

Then select **Kernel → Restart & Run All**.

The notebook will:
1. Load and preprocess the dataset
2. Split by patient ID to prevent leakage (80% train / 15% test / 5% unseen)
3. Apply SMOTENC oversampling on training patients only → 41,635 samples per class
4. Engineer features: `n_meds_changed`, `n_meds_active`, `cci_score` (Charlson Comorbidity Index)
5. Train AutoGluon (LightGBM, CatBoost, WeightedEnsemble) with a 600s time budget
6. Evaluate on hold-out test set with threshold tuning and SHAP explainability

---

## Running Inferencing

To run predictions on the unseen data:

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

def compute_cci(row):
    score = 0
    for col in ['diag_1', 'diag_2', 'diag_3']:
        code = str(row.get(col, '') or '').strip()
        try: n = float(code)
        except: continue
        if 410 <= n <= 411: score += 1
        if n == 428: score += 1
        if 250 <= n <= 250.3: score += 1
        if 582 <= n <= 586: score += 2
        if 140 <= n <= 172.9: score += 2
        if 572 <= n <= 572.9: score += 3
        if 196 <= n <= 199.9: score += 6
        if n == 42: score += 6
    return score

predictor = TabularPredictor.load("ag_models")
unseen    = pd.read_csv("unseen_data.csv")

MED_COLS = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
            'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
            'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
            'insulin','glyburide-metformin','glipizide-metformin',
            'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

med_cols = [c for c in MED_COLS if c in unseen.columns]
unseen['n_meds_changed'] = unseen[med_cols].isin(['Up', 'Down']).sum(axis=1)
unseen['n_meds_active']  = unseen[med_cols].isin(['Steady', 'Up', 'Down']).sum(axis=1)
unseen['cci_score']      = unseen.apply(compute_cci, axis=1)

predictions = predictor.predict(unseen)
print(predictions)
```

Or simply run all cells in `Gruve_SieunPark.ipynb` — the unseen data split is already handled in Section 2.

---

## Model Performance

| Model | Val ROC-AUC | Test ROC-AUC |
|---|---|---|
| WeightedEnsemble_L2 (CatBoost 54.5% + LightGBM 45.5%) | **0.8710** | 0.6551 |
| LightGBM | 0.8596 | 0.6341 |
| CatBoost | 0.8507 | 0.6560 |

**Hold-out test set (19,341 patients) — WeightedEnsemble_L2:**

| Metric | Before Tuning | After Threshold Tuning (t=0.14) |
|---|---|---|
| Accuracy | 57.85% | 48.83% |
| F1 Macro | 0.4445 | 0.4208 |
| F1 `<30` | 0.16 (precision 0.24, recall 0.12) | **0.26 (precision 0.17, recall 0.47)** |
| F1 `>30` | 0.48 (precision 0.50, recall 0.46) | 0.36 (precision 0.50, recall 0.28) |
| F1 `NO` | 0.70 (precision 0.65, recall 0.75) | 0.65 (precision 0.67, recall 0.63) |

Threshold tuning at 0.14 sacrifices overall accuracy to substantially improve recall for the clinically critical `<30` class (early readmission).

**Key EDA findings:**
- `number_inpatient` is the strongest predictor (Kruskal-Wallis H=4,456.5, p≈0)
- `discharge_disposition_id` is the strongest categorical predictor (Chi-squared=2,805.8, p≈0)
- Mean CCI score is highest for `<30` patients (1.353 vs 1.184 for NO), confirming higher comorbidity burden increases early readmission risk
- No strong pairwise correlations (|r| > 0.5) among numerical features

---

## Dependencies

- Python 3.12.4
- autogluon == 1.5.0
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn
- shap, umap-learn

See `requirements.txt` for full version list.
