import torch
import torch.nn as nn
import numpy as np

# Set fixed seed for perfect reproducibility across Python and Aero
torch.manual_seed(42)

print("============================================================================")
print("PYTORCH AUTODIFF BASELINE: 2-Layer MLP (10 Iterations)")
print("============================================================================")

# 1. Network Topology (matching neural_network.aero shapes)
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.01

# 2. Network Parameters (Explicit Initialization)
W1 = torch.tensor([[ 0.5, -0.4],
                   [-0.2,  0.6],
                   [ 0.8, -0.1]], requires_grad=True)
               
b1 = torch.tensor([0.1, -0.1, 0.0], requires_grad=True)

W2 = torch.tensor([[ 0.7, -0.5,  0.3]], requires_grad=True)
b2 = torch.tensor([-0.2], requires_grad=True)

# 3. Training Data (Synthetic)
# We use a batch size of 1 for simplicity in mapping to our Aero prototype
X = torch.tensor([[2.0, 3.0]])
target = torch.tensor([[1.0]])

# 4. Training Loop
for step in range(1, 11):
    print(f"\n--- STEP {step} ---")
    
    # FORWARD PASS
    # Z1 = X @ W1^T + b1
    Z1 = torch.matmul(X, W1.t()) + b1
    # A1 = ReLU(Z1)
    A1 = torch.relu(Z1)
    
    # Z2 = A1 @ W2^T + b2
    Z2 = torch.matmul(A1, W2.t()) + b2
    # A2 = Sigmoid(Z2)
    A2 = torch.sigmoid(Z2)
    
    # Loss: Mean Squared Error (MSE)
    loss = torch.mean((A2 - target) ** 2)
    print(f"Loss: {loss.item():.6f}")
    
    # BACKWARD PASS
    loss.backward()
    
    # Print gradients for Aero validation
    print("Gradients:")
    print(f"dW2: {W2.grad.numpy()}")
    print(f"db2: {b2.grad.numpy()}")
    print(f"dW1: \n{W1.grad.numpy()}")
    print(f"db1: {b1.grad.numpy()}")
    
    # OPTIMIZATION (Vanilla SGD)
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad
        
        # Zero gradients after update
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()
