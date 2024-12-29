import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 4  # Random numbers between 0 and 10
Y = 2 * X + np.random.randn(100, 1)  # Y = 2.5X + noise

# Initialize parameters
m = 0  # Initial slope
c = 0  # Initial intercept
learning_rate = 0.01  # Step size for gradient descent
epochs = 1000  # Number of iterations

# Gradient Descent Algorithm
for _ in range(epochs):
    Y_pred = m * X + c  # Predicted Y
    error = Y - Y_pred  # Error
    m_gradient = -2 * np.sum(X * error) / len(X)  # Derivative w.r.t. m
    c_gradient = -2 * np.sum(error) / len(X)  # Derivative w.r.t. c
    m -= learning_rate * m_gradient  # Update m
    c -= learning_rate * c_gradient  # Update c

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Predictions
Y_pred = m * X + c

# Plot results
import matplotlib.pyplot as plt
plt.scatter(X, Y, color="blue", label="Data points")
plt.plot(X, Y_pred, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression From Scratch")
plt.legend()
plt.show()
