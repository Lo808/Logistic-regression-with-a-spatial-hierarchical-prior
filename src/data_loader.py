import pandas as pd
import numpy as np


def load_data(filepath):

    df=pd.read_csv(filepath)
    df['y'] = (df['trump16'] > df['clinton16']).astype(float)
    feature_cols = [
            'white_pct', 'black_pct', 'hispanic_pct', 'foreignborn_pct',
            'age65andolder_pct', 'median_hh_inc', 'clf_unemploy_pct', 
            'lesscollege_pct', 'rural_pct'
        ]
    df = df.dropna(subset=feature_cols + ['y', 'fips']).copy()
    
    # Extract arrays
    X = df[feature_cols].values
    y = df['y'].values
    fips = df['fips'].values # This is L_i, the location identifier
    
    # 3. Standardize features (Crucial for ADVI / optimization stability)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / (X_std + 1e-8)
    
    return X_scaled, y, fips, df

def spatial_train_test_split(X, y, fips, train_ratio=0.8, seed=42):
    """
    Splits the counties 80/20 for training and validation to test 
    spatial interpolation as outlined in the proposal.
    """
    np.random.seed(seed)
    n_locations = len(fips)
    
    # Randomly shuffle the indices to ensure a random spatial holdout
    indices = np.random.permutation(n_locations)
    
    # Calculate the 80% split index
    split_idx = int(n_locations * train_ratio)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    # Create the training set
    X_train, y_train, fips_train = X[train_idx], y[train_idx], fips[train_idx]
    
    # Create the validation set
    X_val, y_val, fips_val = X[val_idx], y[val_idx], fips[val_idx]
    
    return (X_train, y_train, fips_train), (X_val, y_val, fips_val)

if __name__ == "__main__":
    # Example usage:
    # Replace 'election_data.csv' with your actual file path
    csv_path = r'data\raw\election-context-2018.csv' 
    
    
    # Load and preprocess
    X, y, fips, raw_df = load_data(csv_path)
    print(f"Total counties loaded: {len(y)}")
    
    # Split the data
    train_data, val_data = spatial_train_test_split(X, y, fips)
    
    X_train, y_train, fips_train = train_data
    X_val, y_val, fips_val = val_data
    
    print(f"Training counties (80%): {len(y_train)}")
    print(f"Validation counties (20%): {len(y_val)}")