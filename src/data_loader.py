import pandas as pd
import torch

def load_and_preprocess_election_data(filepath):
    """
    Loads the 1988 Gelman & Hill election polling dataset.
    Extracts features, targets, and builds the nested state/region hierarchy.
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # The standard columns for this specific hierarchical model
    # We drop any rows with missing data in these specific columns
    expected_cols = ['bush', 'state', 'region', 'age', 'edu', 'black', 'female']
    df = df.dropna(subset=expected_cols).copy()
    
    # 1. Target Variable (y)
    # 1 if they voted for Bush, 0 otherwise
    y = torch.tensor(df['bush'].values, dtype=torch.float64)
    
    # 2. Features (X)
    feature_cols = ['age', 'edu', 'black', 'female']
    X_raw = torch.tensor(df[feature_cols].values, dtype=torch.float64)
    
    # Standardize the features for stable gradients in ADVI
    X_mean = X_raw.mean(dim=0)
    X_std = X_raw.std(dim=0)
    X = (X_raw - X_mean) / (X_std + 1e-8)
    
    # 3. State Index (state_idx)
    # The raw data usually has states as 1-51. We need 0-50 for PyTorch indexing.
    # We map whatever unique IDs exist to a strict 0 to n_states-1 range.
    df['state_code'] = df['state'].astype('category').cat.codes
    state_idx = torch.tensor(df['state_code'].values, dtype=torch.long)
    
    # 4. Region Index and state_to_region_idx mapping
    # Regions are usually 1-5. We need 0-4.
    df['region_code'] = df['region'].astype('category').cat.codes
    
    # We need an array where the index is the state_code, and the value is the region_code.
    # This allows the prior to look up the correct regional mean for each state.
    state_region_map = df.groupby('state_code')['region_code'].first().sort_index()
    state_to_region_idx = torch.tensor(state_region_map.values, dtype=torch.long)
    
    return X, y, state_idx, state_to_region_idx

