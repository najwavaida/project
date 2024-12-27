# Importing necessary libraries for interpolation and evaluation
from scipy.interpolate import lagrange, CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Defining the data points
x_data = np.array([-13.8, -8.7, -5.5, 2, 17, 27.3, 31.9], dtype=float)
y_data = np.array([4455, 4125, 3755, 2500, 2080, 1689, 1230], dtype=float)

# Lagrange Interpolation
lagrange_interpolator = lagrange(x_data, y_data)

# Cubic Spline Fit
spline = CubicSpline(x_data, y_data)

# Newton Interpolation
# Creating divided differences for Newton's method
n = len(x_data)
coefficients = np.zeros(n)
coefficients[0] = y_data[0]

for j in range(1, n):
    for i in range(n - 1, j - 1, -1):
        y_data[i] = (y_data[i] - y_data[i - 1]) / (x_data[i] - x_data[i - j])
    coefficients[j] = y_data[j]

# Function for Newton's interpolation
def newton_interpolation(x):
    result = coefficients[0]
    for i in range(1, n):
        term = coefficients[i]
        for j in range(i):
            term *= (x - x_data[j])
        result += term
    return result

# Target x value
x_target = 10

# Interpolated values
x_values = np.linspace(min(x_data), max(x_data), 100)
lagrange_values = lagrange_interpolator(x_values)
spline_values = spline(x_values)
newton_values = [newton_interpolation(x) for x in x_values]

# Interpolated value at x_target
interpolated_value = newton_interpolation(x_target)

# Calculate predictions for evaluation
predictions_lagrange = lagrange_interpolator(x_data)
predictions_spline = spline(x_data)
predictions_newton = [newton_interpolation(x) for x in x_data]

# Error metrics for each method
metrics = {}
for method, predictions in zip(['Lagrange', 'Spline', 'Newton'], [predictions_lagrange, predictions_spline, predictions_newton]):
    rse = np.sqrt(np.sum((y_data - predictions) ** 2) / len(y_data))
    rss = np.sum((y_data - predictions) ** 2)
    tss = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = r2_score(y_data, predictions)
    r = np.sqrt(r2) if r2 >= 0 else float('nan')
    mse = mean_squared_error(y_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_data, predictions)
    mape = np.mean(np.abs((y_data - predictions) / y_data)) * 100
    metrics[method] = {'RSE': rse, 'RSS': rss, 'TSS': tss, 'R^2': r2, 'R': r, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# Plotting the results
plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_values, lagrange_values, label='Lagrange Interpolation', linestyle='--')
plt.plot(x_values, spline_values, label='Cubic Spline Fit', linestyle=':')
plt.plot(x_values, newton_values, label='Newton Interpolation', linestyle='-.')
plt.axhline(y=interpolated_value, color='blue', linestyle='--', label='Interpolated Value at x=10')
plt.title('Comparison of Interpolation Methods')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# Displaying metrics
for method, values in metrics.items():
    print(f"Metrics for {method} Interpolation:")
    for metric, value in values.items():
        print(f"{metric}: {value}")
    print()
