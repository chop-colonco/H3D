import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

print("Starting Training...\n")

# Load data as double
data = pd.read_csv('training_data.csv').values
X = torch.tensor(data[:-20, :-1]).double()
y = torch.tensor(data[:-20, -1]).double().unsqueeze(1)

x_test = torch.tensor(data[-20:, :-1]).double()
y_test = torch.tensor(data[-20:, -1]).double().unsqueeze(1)

model = nn.Sequential(
    nn.Linear(26, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1) 
)

model = model.double()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        mse = nn.functional.mse_loss(y_pred, y).item()
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, MSE: {mse:.6f}")

# Save model parameters
torch.save(model.state_dict(), 'spinflip_model_v2.pth')
print("Training complete. Model saved to 'spinflip_model.pth'\n")

# Test the model
model.eval()
with torch.no_grad():
    y_test_pred = model(x_test)
    test_mse = nn.functional.mse_loss(y_test_pred, y_test).item()
    print(f"\nTest MSE: {test_mse:.6f}")

    # Print predictions vs actual values
    print("\nPredictions vs Actual:")
    for pred, actual in zip(y_test_pred, y_test):
        print(f"Predicted: {pred.item():.3f}, Actual: {actual.item():.3f}")

# Reload model to be safe before scripting
model.load_state_dict(torch.load('spinflip_model_v2.pth'))
model = model.double()
model.eval()

# Convert to TorchScript for C++
scripted_model = torch.jit.script(model)
scripted_model.save('spinflip_model_v2.pt')
print("TorchScript model saved to 'spinflip_model.pt'.")
