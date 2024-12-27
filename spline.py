import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_error(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)  # Residual Sum of Squares
    tss = np.sum((y_true - np.mean(y_true)) ** 2)  # Total Sum of Squares
    r_squared = 1 - (rss / tss) if tss != 0 else float('nan')
    rse = np.sqrt(rss / len(y_true)) if len(y_true) > 0 else float('nan')
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('nan')
    r = np.sqrt(r_squared) if r_squared >= 0 else float('nan')  # Prevent invalid sqrt

    return {
        "RSS": rss,
        "TSS": tss,
        "RSE": rse,
        "R^2": r_squared,
        "R": r,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "MAE": mae
    }

# Example data
x_data = np.array([-13.8, -8.7, -5.5, 2, 17, 27.3, 31.9])
y_data = np.array([4455, 4125, 3755, 2500, 2080, 1689, 1230])

# Generate x_fit for interpolation
x_fit = np.linspace(min(x_data), max(x_data), 500)

# Linear spline interpolation
cs_linear = CubicSpline(x_data[:2], y_data[:2])
y_fit_linear = cs_linear(x_fit)

# Quadratic spline interpolation
cs_quadratic = CubicSpline(x_data[:3], y_data[:3])
y_fit_quadratic = cs_quadratic(x_fit)

# Cubic spline interpolation
cs_cubic = CubicSpline(x_data, y_data)
y_fit_cubic = cs_cubic(x_fit)

# Evaluate error
y_pred_linear = cs_linear(x_data[:2])
y_pred_quadratic = cs_quadratic(x_data[:3])
y_pred_cubic = cs_cubic(x_data)

error_metrics = {
    "Linear Spline": evaluate_error(y_data[:2], y_pred_linear),
    "Quadratic Spline": evaluate_error(y_data[:3], y_pred_quadratic),
    "Cubic Spline": evaluate_error(y_data, y_pred_cubic)
}

# Plot data and spline fits
plt.scatter(x_data, y_data, label="Data Points", color="black")
plt.plot(x_fit, y_fit_linear, label="Linear Spline Fit", linestyle="--")
plt.plot(x_fit, y_fit_quadratic, label="Quadratic Spline Fit", linestyle="-.")
plt.plot(x_fit, y_fit_cubic, label="Cubic Spline Fit", linestyle="-")
plt.title("Spline Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# Print error metrics
for degree, metrics in error_metrics.items():
    print(f"\n{degree} Fit Error Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
