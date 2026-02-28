import numpy as np

def relu(x):
    return np.maximum(0, x)

def step(x):
    return np.where(x > 0, 1, 0)

print("============================================================================")
print("PYTHON BASELINE: Multi-Layer Perceptron Verification")
print("============================================================================")

# 1. Inputs
X = np.array([2, 3])

# 2. Hidden Layer Parameters
W1 = np.array([
    [5, -4],
    [-2, 6],
    [8, -1]
])
# Our aero code was functionally transposed per connection:
# Z1_1 = 2*5 + 3*-4 + 1
# Z1_2 = 2*-2 + 3*6 - 1
# Z1_3 = 2*8 + 3*-1 + 0

b1 = np.array([1, -1, 0])

# 3. Output Layer Parameters
W2 = np.array([7, -5, 3])
b2 = np.array([-2])

# --- FORWARD PASS ---
print("\n--- HIDDEN LAYER COMPUTATION ---")
Z1 = np.dot(W1, X) + b1
print(f"Pre-activation (Z1): {Z1}")

A1 = relu(Z1)
print(f"Activation ReLU (A1): {A1}")

print("\n--- OUTPUT LAYER COMPUTATION ---")
Z2 = np.dot(W2, A1) + b2
print(f"Pre-activation (Z2): {Z2}")

# Sigmoid approximation (Binary Step)
A2 = step(Z2)
print(f"Final Model Prediction (A2): {A2[0]}")

print("\nValidation constraints:")
print(f"Z1 matches Aero? {np.array_equal(Z1, np.array([-1, 13, 13]))}")
print(f"A1 matches Aero? {np.array_equal(A1, np.array([0, 13, 13]))}")
print(f"Z2 matches Aero? {Z2[0] == -28}")
print(f"A2 matches Aero? {A2[0] == 0}")
