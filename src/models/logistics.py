import torch

def compute_log_likelihood(X, y, state_idx, beta, alpha):
    """
    Computes the log likelihood of the observed data given the parameters.
    """
    # Calculate the linear predictor
    # X_i * beta + alpha_{state[i]}
    linear_predictor = torch.matmul(X, beta) + alpha[state_idx]
    
    # We use PyTorch's built-in binary cross entropy with logits 
    # which is numerically stable and exactly equivalent to the negative 
    # Bernoulli log-likelihood. We negate it to get the positive log-likelihood.
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    
    # BCE returns negative log likelihood, so we return the negative of that
    log_likelihood = -loss_fn(linear_predictor, y)
    
    return log_likelihood