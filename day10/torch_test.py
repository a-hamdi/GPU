import torch
import torch.nn.functional as F

# Define Q, K, V, and expected O matrices
def load_csv_to_tensor(file_path):
    df = pd.read_csv(file_path, header=None)  # Assuming no header in the CSV
    tensor = torch.tensor(df.values)  # Convert DataFrame to tensor
    return tensor
# Load the matrices
Q = load_csv_to_tensor('query_output.csv')
K = load_csv_to_tensor('key_output.csv')
V = load_csv_to_tensor('value_output.csv')
expected_O = load_csv_to_tensor('output_output.csv')
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
