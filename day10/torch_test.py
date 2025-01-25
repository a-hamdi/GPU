import torch
import torch.nn.functional as F

# Define Q, K, V, and expected O matrices (from the main)
Q = torch.tensor([[0.0853815, 0.250728],
                  [0.0798073, 0.585853]], dtype=torch.float32)

K = torch.tensor([[0.0853815, 0.250728],
                  [0.0798073, 0.585853]], dtype=torch.float32)

V = torch.tensor([[0.0853815, 0.250728],
                  [0.0798073, 0.585853]], dtype=torch.float32)

expected_O = torch.tensor([[2.74403e-08, 2.80995e+37],
                           [8.29159e+23, 1.72871e-12]], dtype=torch.float32)

# Step 1: Scaled Dot-Product Attention Calculation
# Compute the scaled dot-product of Q and K^T
dk = Q.size(-1)  # Dimension of the embedding (key size)
scores = torch.matmul(Q, K.T) / (dk ** 0.5)  # Scaling factor by sqrt(d_k)

# To prevent instability, apply softmax in a numerically stable way
attention_weights = F.softmax(scores, dim=-1)

# Step 2: Multiply attention weights with V to compute O
computed_O = torch.matmul(attention_weights, V)

# Step 3: Compare computed_O with expected_O
is_close = torch.allclose(computed_O, expected_O, atol=1e-5)

# Print results
print("Attention Weights:")
print(attention_weights)
print("Computed Output O:")
print(computed_O)
print("Expected Output O:")
print(expected_O)
print("Do the computed output and expected output match (within tolerance)?", is_close)

# Debugging: Exploring potential issues in scores
print("Raw scores (Q @ K.T):")
print(scores)
print("Max score (for stability):", torch.max(scores))
