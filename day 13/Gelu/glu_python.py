import numpy as np
from scipy.stats import norm

def gelu(x):
    return x * norm.cdf(x)

# Example usage
x = np.array([-1*i/2 for i in range(100)] )
print("GELU:", gelu(x))
