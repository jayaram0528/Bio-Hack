# Drug Biomarker Prediction

This project predicts drug response (biomarker values) from molecular data using a machine learning approach based on RDKit molecular descriptors and XGBoost regression. It also provides a user-friendly web interface using Streamlit for interactive predictions.

---

## 📁 File Structure

```
├── drug_biomarker_model.py         # Training script for XGBoost model
├── valid_data_prediction.py        # Prediction script using trained model
├── xgb_trained_model.json          # Trained XGBoost model
├── app.py                         # Streamlit web app for interactive prediction
├── submission.csv                  # Final output with predicted biomarker values
├── train.csv                       # Training dataset (required by training script)
├── valid.csv                       # Validation dataset (required by prediction script)
├── requirements.txt                # Python dependencies
```

---

## 📦 Requirements

Install the required packages:

```bash
pip install -r requirements.txt
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

## 🌐 Streamlit Web App (`app.py`)

The Streamlit app provides an interactive web interface for predicting biomarker values from SMILES strings.

### How to Run

1. Make sure all dependencies are installed (see [Requirements](#requirements)).
2. Run the following command in your project directory:

    ```bash
    streamlit run app.py
    ```

3. A browser window will open. Enter a SMILES string to get the predicted bio marker value.

---

## 📄 Output

The final predictions are saved in:

```
submission.csv
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
- The Streamlit app requires the trained model file (`xgb_trained_model.json`) to be present in the project directory.