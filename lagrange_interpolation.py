# Defining the Lagrange interpolation method
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Defining the x_data and y_data
x_data = np.array([-13.8, -8.7, -5.5, 2, 17, 27.3, 31.9], dtype=float)
y_data = np.array([4455, 4125, 3755, 2500, 2080, 1689, 1230], dtype=float)

# Setting the target x value
x_target = 10

# Lagrange interpolation function
def lagrange_interpolation(x, y, x_target):
    total = 0.0
    n = len(x)
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_target - x[j]) / (x[i] - x[j])
        total += term
    return total

# Calculate interpolated value using Lagrange method
y_lagrange = lagrange_interpolation(x_data, y_data, x_target)

# Fit polynomial models
coeffs_linear = np.polyfit(x_data, y_data, 1)
coeffs_quadratic = np.polyfit(x_data, y_data, 2)
coeffs_cubic = np.polyfit(x_data, y_data, 3)

# Generate polynomial functions
poly_linear = np.poly1d(coeffs_linear)
poly_quadratic = np.poly1d(coeffs_quadratic)
poly_cubic = np.poly1d(coeffs_cubic)

# Calculate predictions for the target x value
y_linear = poly_linear(x_target)
y_quadratic = poly_quadratic(x_target)
y_cubic = poly_cubic(x_target)

# Calculate errors
def calculate_errors():
    y_lagrange_values = np.array([lagrange_interpolation(x_data, y_data, xi) for xi in x_data])
    RSE = np.sqrt(np.sum((y_data - y_lagrange_values) ** 2) / len(y_data))
    MSE = mean_squared_error(y_data, y_lagrange_values)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(y_lagrange_values - y_data))
    MAPE = np.mean(np.abs((y_lagrange_values - y_data) / y_data)) * 100
    RSS = np.sum((y_lagrange_values - y_data) ** 2)
    TSS = np.sum((y_data - np.mean(y_data)) ** 2)
    R2 = r2_score(y_data, y_lagrange_values)
    R = np.sqrt(R2) if R2 >= 0 else float('nan')
    return RSE, MSE, RMSE, MAE, MAPE, RSS, TSS, R2, R

RSE, MSE, RMSE, MAE, MAPE, RSS, TSS, R2, R = calculate_errors()

# Plotting the results
plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.scatter(x_target, y_lagrange, color='blue', label='Lagrange Interpolation at x=10')
plt.plot(x_data, poly_linear(x_data), label='Linear Fit', linestyle='--')
plt.plot(x_data, poly_quadratic(x_data), label='Quadratic Fit', linestyle='-.')
plt.plot(x_data, poly_cubic(x_data), label='Cubic Fit', linestyle=':')
plt.title('Lagrange Interpolation and Polynomial Fits')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# Displaying the results
print('\nLagrange Interpolated Value at x = 10:', y_lagrange)
print('Linear Fit Value at x = 10:', y_linear)
print('Quadratic Fit Value at x = 10:', y_quadratic)
print('Cubic Fit Value at x = 10:', y_cubic)
print('\nLinear Coefficients:', coeffs_linear)
print('Quadratic Coefficients:', coeffs_quadratic)
print('Cubic Coefficients:', coeffs_cubic)
print('\nRSE:', RSE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAE:', MAE)
print('MAPE:', MAPE)
print('R^2:', R2)
print('R:', R)
print('RSS:', RSS)
print('TSS:', TSS)
