import torch
import math

class ADVI_ELBO:

    """ 
    This class contains the logic for estimating the ELBO function using MC integration.
    In order to compute the gradients of the ELBO loss with autograd, we need to reparameterize the sampling process.
    Then e draw samples from a standard normal distribution and transform them using the inverse transormations (variationnal parameters and bijectors).
    The results are unbiased estimates of the ELBO, which we can then use to compute gradients and perform SGA updates"""

    def __init__(self, bijector, prior_fn, likelihood_fn):
        self.bijector = bijector
        self.prior_fn = prior_fn
        self.likelihood_fn = likelihood_fn

    def compute_loss(self, X, y, state_idx, state_to_region_idx, variational_params, n_samples=1):
        """
        Computes the Negative ELBO gradients using Monte Carlo integration.
        n_samples corresponds to S in the proposal.
        """
        elbo_estimates = []
        
        for _ in range(n_samples):
            # 1. Reparameterization Trick: Sample from q(eta; phi)
            # eta = mu + sigma * epsilon, where epsilon ~ N(0, 1)
            samples = {}
            entropy = 0.0
            
            for param_name, (mu, log_sigma) in variational_params.items():
                sigma = torch.exp(log_sigma)
                epsilon = torch.randn_like(mu)
                eta = mu + sigma * epsilon
                samples[param_name] = eta
                
                # Analytical entropy of a Gaussian: sum(log(sigma)) + const
                entropy += torch.sum(log_sigma) + 0.5 * mu.numel() * (1.0 + math.log(2 * math.pi))

            # 2. Coordinate Transformation (for strictly positive variance params)
            # Map unconstrained real space to positive space and get Jacobians
            sigma_state_real = samples['sigma_state']
            sigma_region_real = samples['sigma_region']
            
            sigma_state = self.bijector.forward(sigma_state_real)
            sigma_region = self.bijector.forward(sigma_region_real)
            
            log_det_jacobian = self.bijector.log_abs_det_jacobian(sigma_state_real).sum() + \
                               self.bijector.log_abs_det_jacobian(sigma_region_real).sum()

            # 3. Compute Log Prior p(eta)
            log_prior = self.prior_fn(
                beta=samples['beta'], 
                alpha=samples['alpha'], 
                gamma=samples['gamma'], 
                sigma_state=sigma_state, 
                sigma_region=sigma_region, 
                state_to_region_idx=state_to_region_idx
            )
            
            # 4. Compute Log Likelihood p(x | eta)
            log_likelihood = self.likelihood_fn(
                X=X, 
                y=y, 
                state_idx=state_idx, 
                beta=samples['beta'], 
                alpha=samples['alpha']
            )
            
            # 5. Combine to form the ELBO sample
            # ELBO = E_q[log p(x, eta)] + H(q) + log_det_jacobian
            joint_log_prob = log_likelihood + log_prior + log_det_jacobian
            elbo_sample = joint_log_prob + entropy
            elbo_estimates.append(elbo_sample)
            
        # Average over S samples
        avg_elbo = torch.stack(elbo_estimates).mean()
        
        # We minimize the *Negative* ELBO
        return -avg_elbo