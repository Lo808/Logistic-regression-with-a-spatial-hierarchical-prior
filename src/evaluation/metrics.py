import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_predictive_metrics(X, y, state_idx, vp, n_samples=1000):
    """
    Computes Accuracy, AUC, and Log-Predictive Density (LPD) 
    using samples from the learned variational posterior.
    """

    with torch.no_grad():
        # 1. Extract learned parameters for beta (global) and alpha (state-level)
        beta_mu, beta_log_sigma = vp['beta']
        alpha_mu, alpha_log_sigma = vp['alpha']
        
        beta_sigma = torch.exp(beta_log_sigma)
        alpha_sigma = torch.exp(alpha_log_sigma)
        
        # 2. Draw S samples from the variational posterior q(eta; phi)
        # beta_samples shape: (n_samples, n_features)
        beta_samples = beta_mu + beta_sigma * torch.randn((n_samples, len(beta_mu)), dtype=torch.float64)
        
        # alpha_samples shape: (n_samples, n_states)
        alpha_samples = alpha_mu + alpha_sigma * torch.randn((n_samples, len(alpha_mu)), dtype=torch.float64)
        
        # 3. Compute probabilities for each sample simultaneously via matrix math
        # X shape: (N, n_features), beta_samples.T shape: (n_features, n_samples)
        linear_predictor = torch.matmul(X, beta_samples.T) # Shape: (N, n_samples)
        
        # Map the state intercepts to the respective voters and add them
        # alpha_samples[:, state_idx] is (n_samples, N), so we transpose it to (N, n_samples)
        state_intercepts = alpha_samples[:, state_idx].T
        
        logits = linear_predictor + state_intercepts
        
        # Apply inverse-logit (sigmoid) to get probabilities
        probs = torch.sigmoid(logits) # Shape: (N, n_samples)
        
        # 4. Bayesian Model Averaging
        # Average the predicted probabilities across all samples
        mean_probs = probs.mean(dim=1).numpy()
        y_np = y.numpy()
        
        # 5. Calculate Standard Classification Metrics
        preds = (mean_probs > 0.5).astype(int)
        accuracy = accuracy_score(y_np, preds)
        auc = roc_auc_score(y_np, mean_probs)
        
        # 6. Calculate Log-Predictive Density (LPD)
        # If true y=1, likelihood is prob. If true y=0, likelihood is 1 - prob.
        p_y = torch.where(y.unsqueeze(1) == 1, probs, 1.0 - probs)
        
        # Average over the samples (dim=1), take the log, then sum over all voters
        lpd = torch.log(p_y.mean(dim=1)).sum().item()
        
        return accuracy, auc, lpd