import math
import torch
import triton
import triton.language as tl

# ----------------------------
# Triton kernel for the objective function
# ----------------------------
@triton.jit
def objective_function_kernel(x_ptr, result_ptr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    x = tl.load(x_ptr + pid)

    res = (x - 3.0) * (x - 3.0)

    tl.store(result_ptr + pid, res)

def objective_function(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)

    BLOCK_SIZE = 1
   
    grid = lambda meta: (x.numel(),)
    objective_function_kernel[grid](x, result, BLOCK_SIZE=BLOCK_SIZE)
    return result

# ----------------------------
# Simulated Annealing (SA) implementation in Python
# ----------------------------
def simulated_annealing_sa(num_steps=1000, initial_x=0.0, initial_temp=10.0, alpha=0.99):
    
    x = initial_x
    
    current_val = objective_function(torch.tensor([x], device='cuda', dtype=torch.float32)).item()
    best_x = x
    best_val = current_val
    temp = initial_temp

    for step in range(num_steps):
        candidate = x + (torch.rand(1).item() - 0.5) * 2.0  
        candidate_val = objective_function(torch.tensor([candidate], device='cuda', dtype=torch.float32)).item()
        delta = candidate_val - current_val

        if delta < 0 or torch.rand(1).item() < math.exp(-delta / temp):
            x = candidate
            current_val = candidate_val
            if candidate_val < best_val:
                best_val = candidate_val
                best_x = candidate

        temp *= alpha

    return best_x, best_val

if __name__ == '__main__':
    torch.cuda.manual_seed(0)  
    best_solution, best_objective = simulated_annealing_sa()
    print("Best solution found: x = {:.4f} with objective value = {:.4f}"
          .format(best_solution, best_objective))
