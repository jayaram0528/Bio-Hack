# importing necessary libraries
import pandas as pd 
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# read the training data
train_data = pd.read_csv('train.csv')

## Function to convert SMILES to Morgan fingerprint
def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    """
    Convert a SMILES string to a Morgan fingerprint.

    Args:
        smiles (str): The SMILES representation of the molecule.
        radius (int, optional): The radius parameter for the Morgan fingerprint. Default is 2.
        n_bits (int, optional): The length of the fingerprint vector. Default is 2048.

    Returns:
        list: A list of bits representing the Morgan fingerprint, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)

def compute_descriptors(smiles):
    """
    Compute molecular descriptors for a given SMILES string.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        list: A list containing [MolWt, LogP, TPSA, HBD, HBA] or [None]*5 if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None]*5  # Adjust if you add more descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return [mw, logp, tpsa, hbd, hba]

# Apply to your dataframe
descriptor_names = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA']
descriptor_values = train_data['Drug'].apply(compute_descriptors)
# Convert to DataFrame using the computed descriptors and assign column names
desc_df = pd.DataFrame(descriptor_values.tolist(), columns=descriptor_names)
# Concatenate with original data
train_data = pd.concat([train_data, desc_df], axis=1)
# Convert SMILES to Morgan fingerprints
train_data['morgan_fp'] = train_data['Drug'].apply(smiles_to_morgan_fp)


# Prepare features and target variable
# Assuming 'Bio_Marker_Value' is the target variable 
# input features are the Morgan fingerprints and descriptors
desc_names = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA']
X_fp = np.array(train_data['morgan_fp'].tolist())
X_desc = train_data[desc_names].values

features = np.concatenate([X_fp, X_desc], axis=1)
target = train_data['Bio_Marker_Value'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train an XGBoost model
xgb_model = XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=3.0,
    random_state=42
)
# Fit the model on the training data
xgb_model.fit(X_train, y_train)

# make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)
# evaluate the model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f'XGBoost Mean Absolute Error: {mae_xgb}')
print(f'XGBoost R2 Score: {r2_xgb}')
mae_train_xgb = mean_absolute_error(y_train, xgb_model.predict(X_train))

# evaluate the model on the training set
r2_train_xgb = r2_score(y_train, xgb_model.predict(X_train))
print(f'Train XGBoost Mean Absolute Error: {mae_train_xgb}')
print(f'Train XGBoost R2 Score: {r2_train_xgb}')

# Save the model to use later
# Save the model to a file
xgb_model.save_model("xgb_trained_model_1.json")
print("Model saved to xgb_trained_model.json")





