from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd 
import numpy as np
from rdkit.Chem import Descriptors

xgb_model = XGBRegressor()
xgb_model.load_model("xgb_trained_model.json")

# function to construct Morgan fingerprints from SMILES

def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
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



# define a function to predict the drug response
def predict_drug_response(csv_file_path):
    """
    Predict drug response based on the provided CSV file.

    Args:
        csv_file_path (str): The path to the CSV file containing drug data.

    Returns:
        pd.DataFrame: A DataFrame with predictions and original data.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Convert SMILES to Morgan fingerprints
    data['morgan_fp'] = data['Drug'].apply(smiles_to_morgan_fp)

    # Compute molecular descriptors
    descriptor_names = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA']
    descriptor_values = data['Drug'].apply(compute_descriptors)
    
    # Convert to DataFrame using the computed descriptors and assign column names
    desc_df = pd.DataFrame(descriptor_values.tolist(), columns=descriptor_names)
    
    # Concatenate with original data
    data = pd.concat([data, desc_df], axis=1)

    # Prepare features for prediction
    # my input features are the Morgan fingerprints and descriptors
    desc_names = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA']
    X_fp = np.array(data['morgan_fp'].tolist())
    desc_features = data[desc_names].values
    features = np.hstack((X_fp, desc_features))
    # Ensure features are in the correct shape
    if features.ndim == 1:
        features = features.reshape(1, -1)
    # Make predictions
    predictions = xgb_model.predict(features)

    # Add predictions to the DataFrame
    data['Bio_Marker_Value'] = predictions

    # from the dataframe,write Drug_ID,Drug,Bio_Marker_Value into a new csv file
    output_df = data[['Drug_ID', 'Drug', 'Bio_Marker_Value']]
    output_df.to_csv('predicted_biomarker_values.csv', index=False)
    print("Predictions saved to 'predicted_biomarker_values.csv'")

    return output_df
    

# Example usage
if __name__ == "__main__":
    csv_file_path = 'valid.csv'  # Replace with your actual CSV file path
    predictions_df = predict_drug_response(csv_file_path)
