import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

print("============================================================================")
print("PYTORCH AERONN BASELINE: 10 Epochs Modular Training")
print("============================================================================")

class AeroNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 Features -> 3 Hidden -> 1 Output (matching Phase 8.1)
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

        # Initialize specific weights to match previous test
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[ 0.5, -0.4],
                                                [-0.2,  0.6],
                                                [ 0.8, -0.1]]))
            self.fc1.bias.copy_(torch.tensor([0.1, -0.1, 0.0]))
            
            self.fc2.weight.copy_(torch.tensor([[ 0.7, -0.5,  0.3]]))
            self.fc2.bias.copy_(torch.tensor([-0.2]))

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = AeroNNBaseline()
criterion = nn.MSELoss()

# We test with SGD first per the prompt constraints
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Input data
X = torch.tensor([[2.0, 3.0]])
y = torch.tensor([[1.0]])

print("Initial Network:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.numpy()}")

print("\n--- TRAINING LOOP (10 EPOCHS) ---")
for epoch in range(1, 11):
    optimizer.zero_grad()
    
    # Forward Pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward Pass
    loss.backward()
    
    # Update Weights
    optimizer.step()
    
    print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

print("\n--- FINAL PARAMETERS ---")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.numpy()}")
