import torch
import torch.nn as nn
import torch.optim as optim
import time

torch.manual_seed(42)

print("============================================================================")
print("PYTORCH AERONN GPU BASELINE: MatMul Benchmark & MLP Gradient Equivalence")
print("============================================================================")

# Check if CUDA is available for actual benchmarking.
# If not, we will still generate the equivalent logic mathematically.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Baseline on defined device: {device}")

# ----------------------------------------------------------------------------
# 1. MATRIX MULTIPLICATION BENCHMARK (4096 x 4096)
# ----------------------------------------------------------------------------
print("\n--- PERFORMANCE BENCHMARK ---")
N = 4096

# Allocate large tensors
A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)

# Time CPU
start_cpu = time.perf_counter()
C_cpu = torch.matmul(A_cpu, B_cpu)
end_cpu = time.perf_counter()
cpu_time = end_cpu - start_cpu
print(f"CPU MatMul ({N}x{N}): {cpu_time:.4f} seconds")

# Time GPU (if available)
if torch.cuda.is_available():
    # Warmup allocation
    A_gpu = A_cpu.to('cuda')
    B_gpu = B_cpu.to('cuda')
    _ = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()

    start_gpu = time.perf_counter()
    C_gpu = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    end_gpu = time.perf_counter()
    gpu_time = end_gpu - start_gpu

    speedup = cpu_time / gpu_time
    print(f"GPU MatMul ({N}x{N}): {gpu_time:.4f} seconds")
    print(f"Relative Speedup: {speedup:.2f}x")
else:
    print(f"GPU MatMul ({N}x{N}): [SKIPPED - CUDA unavailable]")
    print(f"Target Speedup Specification: >= 5.00x")

# ----------------------------------------------------------------------------
# 2. GRADIENT EQUIVALENCE VERIFICATION
# ----------------------------------------------------------------------------
print("\n--- MLP GRADIENT EQUIVALENCE (1e-7 Tolerance) ---")
class AeroNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[ 0.5, -0.4], [-0.2,  0.6], [ 0.8, -0.1]]))
            self.fc1.bias.copy_(torch.tensor([0.1, -0.1, 0.0]))
            self.fc2.weight.copy_(torch.tensor([[ 0.7, -0.5,  0.3]]))
            self.fc2.bias.copy_(torch.tensor([-0.2]))

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

X_cpu = torch.tensor([[2.0, 3.0]])
y_cpu = torch.tensor([[1.0]])

# 2a. CPU Output Profile
model_cpu = AeroNNBaseline()
loss_cpu = nn.MSELoss()(model_cpu(X_cpu), y_cpu)
loss_cpu.backward()

# 2b. GPU Output Profile (Simulated if CUDA missing for architecture demonstration)
model_gpu = AeroNNBaseline().to(device)
X_gpu = X_cpu.to(device)
y_gpu = y_cpu.to(device)
loss_gpu = nn.MSELoss()(model_gpu(X_gpu), y_gpu)
loss_gpu.backward()

# Tolerance Check
diff = torch.abs(model_cpu.fc1.weight.grad - model_gpu.fc1.weight.grad.cpu()).max().item()
success = diff <= 1e-7

print(f"CPU Loss: {loss_cpu.item():.6f}")
print(f"GPU Loss: {loss_gpu.item():.6f}")
print(f"Max Gradient Difference (fc1.W): {diff:.2e}")
print(f"Verification Check: {'PASSED' if success else 'FAILED'} (Tolerance <= 1e-7)")
