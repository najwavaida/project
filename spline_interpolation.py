# Importing necessary libraries for spline interpolation
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Defining the data points
x_data = np.array([-13.8, -8.7, -5.5, 2, 17, 27.3, 31.9], dtype=float)
y_data = np.array([4455, 4125, 3755, 2500, 2080, 1689, 1230], dtype=float)

# Setting the target x value
x_target = 10

# Cubic Spline Interpolation
spline = CubicSpline(x_data, y_data)
y_spline = spline(x_target)

# Calculate spline errors
RSE_spline = np.sqrt(np.sum((y_data - spline(x_data)) ** 2) / len(y_data))
MSE_spline = mean_squared_error(y_data, spline(x_data))
RMSE_spline = np.sqrt(MSE_spline)
MAE_spline = np.mean(np.abs(spline(x_data) - y_data))
MAPE_spline = np.mean(np.abs((spline(x_data) - y_data) / y_data)) * 100
RSS_spline = np.sum((spline(x_data) - y_data) ** 2)
TSS_spline = np.sum((y_data - np.mean(y_data)) ** 2)
R2_spline = r2_score(y_data, spline(x_data))
R_spline = np.sqrt(R2_spline)

# Plotting the results with spline
plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_data, spline(x_data), label='Cubic Spline Fit', linestyle='--', color='purple')
plt.scatter(x_target, y_spline, color='blue', label='Spline Interpolation at x=10')
plt.title('Cubic Spline Interpolation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# Displaying results
print('Spline Interpolated Value at x = 10:', y_spline)
print('RSE (Spline):', RSE_spline)
print('MSE (Spline):', MSE_spline)
print('RMSE (Spline):', RMSE_spline)
print('MAE (Spline):', MAE_spline)
print('MAPE (Spline):', MAPE_spline)
print('R^2 (Spline):', R2_spline)
print('R (Spline):', R_spline)
print('RSS (Spline):', RSS_spline)
print('TSS (Spline):', TSS_spline)
