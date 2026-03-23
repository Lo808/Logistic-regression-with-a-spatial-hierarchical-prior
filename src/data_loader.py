import pandas as pd
import torch

def load_and_preprocess_election_data(filepath=r'data\raw\polls.subset.dat'):
    """
    Loads the 1988 Gelman & Hill election polling dataset.
    Extracts features, targets, and builds the nested state/region hierarchy.
    """
    df = pd.read_csv(filepath, sep=r'\s+')
    
    # 1. Création de la cartographie des régions
    # Gelman divise les USA en 5 régions (1=Nord-Est, 2=Midwest, 3=Sud, 4=Ouest, 5=D.C.)
    # Ce dictionnaire associe l'ID de chaque État (1-51) à sa région correspondante.
    region_mapping = {
        1:3, 2:4, 3:4, 4:3, 5:4, 6:4, 7:1, 8:1, 9:5, 10:3,
        11:3, 12:4, 13:4, 14:2, 15:2, 16:2, 17:2, 18:3, 19:3, 20:1,
        21:1, 22:1, 23:2, 24:2, 25:3, 26:2, 27:4, 28:2, 29:4, 30:1,
        31:1, 32:4, 33:1, 34:3, 35:2, 36:3, 37:4, 38:1, 39:1, 40:3,
        41:2, 42:3, 43:3, 44:4, 45:1, 46:3, 47:4, 48:4, 49:3, 50:2, 51:4
    }
    
    # On crée la colonne region manquante !
    df['region'] = df['state'].map(region_mapping)
    
    # On s'assure qu'il n'y a pas de valeurs manquantes (NA) dans les colonnes qui nous intéressent
    expected_cols = ['bush', 'state', 'region', 'age', 'edu', 'black', 'female']
    df = df.dropna(subset=expected_cols).copy()
    
    # --- La suite reste identique ---
    # Target
    y = torch.tensor(df['bush'].values, dtype=torch.float64)
    
    # Features (Standardisation)
    feature_cols = ['age', 'edu', 'black', 'female']
    X_raw = torch.tensor(df[feature_cols].values, dtype=torch.float64)
    X_mean = X_raw.mean(dim=0)
    X_std = X_raw.std(dim=0)
    X = (X_raw - X_mean) / (X_std + 1e-8)
    
    # Indexation 0-based pour PyTorch
    df['state_code'] = df['state'].astype('category').cat.codes
    state_idx = torch.tensor(df['state_code'].values, dtype=torch.long)
    
    df['region_code'] = df['region'].astype('category').cat.codes
    state_region_map = df.groupby('state_code')['region_code'].first().sort_index()
    state_to_region_idx = torch.tensor(state_region_map.values, dtype=torch.long)
    
    return X, y, state_idx, state_to_region_idx

