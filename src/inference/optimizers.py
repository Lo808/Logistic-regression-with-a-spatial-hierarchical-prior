import torch

class ADVIOptimizer:
    def __init__(self, n_features, n_states, n_regions, lr=0.1):
        # Initialize variational parameters (phi = {mu, log_sigma})
        # We require gradients for all of these to perform autograd
        dtype = torch.float64
        
        self.vp = {
            'beta': (torch.zeros(n_features, requires_grad=True, dtype=dtype), 
                     torch.zeros(n_features, requires_grad=True, dtype=dtype)),
            'alpha': (torch.zeros(n_states, requires_grad=True, dtype=dtype), 
                      torch.zeros(n_states, requires_grad=True, dtype=dtype)),
            'gamma': (torch.zeros(n_regions, requires_grad=True, dtype=dtype), 
                      torch.zeros(n_regions, requires_grad=True, dtype=dtype)),
            'sigma_state': (torch.zeros(1, requires_grad=True, dtype=dtype), 
                            torch.zeros(1, requires_grad=True, dtype=dtype)),
            'sigma_region': (torch.zeros(1, requires_grad=True, dtype=dtype), 
                             torch.zeros(1, requires_grad=True, dtype=dtype)),
        }
        
        all_params = []
        for mu, log_sigma in self.vp.values():
            all_params.extend([mu, log_sigma])
            
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        
        # Collect all parameters into a single list for the PyTorch optimizer
        all_params = []
        for mu, log_sigma in self.vp.values():
            all_params.extend([mu, log_sigma])
            
        # Using ADAM as suggested for the Stochastic Gradient Ascent
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        
    def step(self, elbo_calculator, X, y, state_idx, state_to_region_idx, n_samples=1):
        """
        Performs a single optimization step.
        """
        self.optimizer.zero_grad()
        
        # Compute the negative ELBO
        loss = elbo_calculator.compute_loss(
            X, y, state_idx, state_to_region_idx, self.vp, n_samples
        )
        
        # Backpropagation via autograd
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()