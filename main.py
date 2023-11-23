import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample Data for Regression
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

# Tkinter GUI
def train_linear_regression():
    # PyTorch Model
    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    epochs = 1000
    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Prediction
    model.eval()
    with torch.no_grad():
        test_inputs = torch.from_numpy(X_test).float()
        predictions = model(test_inputs).numpy()

    # Visualization with Plotly
    fig = px.scatter(x=X_test.flatten(), y=y_test.flatten(), title="Linear Regression Prediction")
    fig.add_trace(px.line(x=X_test.flatten(), y=predictions.flatten(), name='Regression Line').data[0])
    fig.show()

# Tkinter GUI Setup
root = tk.Tk()
root.title("Regression App")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

btn_train = ttk.Button(frame, text="Train Linear Regression", command=train_linear_regression)
btn_train.grid(row=0, column=0, pady=10)

root.mainloop()
