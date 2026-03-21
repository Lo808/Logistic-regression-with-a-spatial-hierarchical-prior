import torch

class ExpBijector:
    """
    Transforms unconstrained real values to strictly positive values.
    Used for variance/standard deviation parameters.
    """
    def forward(self, x):
        # Maps R -> R+
        return torch.exp(x)
    
    def inverse(self, y):
        # Maps R+ -> R
        return torch.log(y)
    
    def log_abs_det_jacobian(self, x):
        # The derivative of exp(x) is exp(x). 
        # The log of the absolute value is just x.
        return x