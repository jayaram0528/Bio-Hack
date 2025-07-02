
# Drug Biomarker Prediction

This project predicts drug response (biomarker values) from molecular data using a machine learning approach based on RDKit molecular descriptors and XGBoost regression.

---

## 📁 File Structure

```
├── drug_biomarker_model.py         # Training script for XGBoost model
├── valid_data_prediction.py       # Prediction script using trained model
├── xgb_trained_model.json         # Trained XGBoost model
├── submission.csv                 # Final output with predicted biomarker values
├── train.csv                      # Training dataset (required by training script)
├── valid.csv                      # Validation dataset (required by prediction script)
```

---

## 📦 Requirements

Install the required packages:

```bash
pip install pandas numpy scikit-learn xgboost rdkit
```

---

## 🧠 Model Training (`drug_biomarker_model.py`)

This script trains an XGBoost regression model using the `train.csv` dataset. It:

- Extracts **Morgan fingerprints** and **molecular descriptors** from SMILES.
- Concatenates them as features.
- Trains an `XGBRegressor`.
- Evaluates the model using MAE and R² on train/test sets.
- Saves the model as `xgb_trained_model.json`.

Run with:

```bash
python drug_biomarker_model.py
```

---

## 🔍 Drug Response Prediction (`valid_data_prediction.py`)

This script predicts drug response values using a trained model. It:

- Loads the trained model from `xgb_trained_model.json`.
- Reads a validation CSV (`valid.csv`) with columns like `Drug_ID` and `Drug`.
- Computes Morgan fingerprints and molecular descriptors.
- Generates predictions for the `Bio_Marker_Value`.
- Saves the output to `predicted_biomarker_values.csv`.

Run with:

```bash
python valid_data_prediction.py
```

---

## 📄 Output

The final predictions are saved in:

```
Submission.csv
```

It contains the columns:

- `Drug_ID`
- `Drug`
- `Bio_Marker_Value` (predicted)

This file can be directly used as the assignment submission.

---

## 📌 Notes

- Ensure `train.csv` and `valid.csv` are present in the same directory before running the scripts.
- The descriptor set used includes:
  - Molecular Weight (MolWt)
  - LogP
  - Topological Polar Surface Area (TPSA)
  - Number of H-Bond Donors (HBD)
  - Number of H-Bond Acceptors (HBA)
