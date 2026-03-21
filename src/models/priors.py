import torch
import math

def normal_log_prob(value, loc, scale):
    """Computes the log probability of a Normal distribution."""
    var = scale ** 2
    log_scale = torch.log(scale)
    return -0.5 * ((value - loc) ** 2 / var) - log_scale - 0.5 * math.log(2 * math.pi)

def compute_hierarchical_prior_log_prob(beta, alpha, gamma, sigma_state, sigma_region, state_to_region_idx):
    """
    Computes the total log prior probability for the hierarchical model.
    """
    # 1. Broad prior on global beta coefficients
    log_p_beta = normal_log_prob(beta, loc=0.0, scale=10.0).sum()
    
    # 2. Regional baselines (gamma)
    log_p_gamma = normal_log_prob(gamma, loc=0.0, scale=sigma_region).sum()
    
    # 3. State baselines (alpha) centered around their respective region's baseline
    # state_to_region_idx maps each state (0-50) to its region (0-4)
    region_means_for_states = gamma[state_to_region_idx]
    log_p_alpha = normal_log_prob(alpha, loc=region_means_for_states, scale=sigma_state).sum()
    
    return log_p_beta + log_p_gamma + log_p_alpha