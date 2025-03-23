"""
Linear Regression with Mathematical Derivation

This module implements linear regression from scratch, deriving the best-fit line
using both:
1. Normal Equations method (closed-form solution)
2. Gradient Descent method (iterative solution)

Mathematical Background:
----------------------
For a linear model y = mx + b:
1. Loss function (Mean Squared Error):
   L(m,b) = (1/n)∑(y_i - (mx_i + b))²

2. Normal Equations solution:
   [m] = (X^T X)^(-1) X^T y
   [b]
   where X is the design matrix with a column of 1's added

3. Gradient Descent updates:
   m = m - α * ∂L/∂m
   b = b - α * ∂L/∂b
   where:
   ∂L/∂m = (-2/n)∑(y_i - (mx_i + b))x_i
   ∂L/∂b = (-2/n)∑(y_i - (mx_i + b))

Real-world applications:
1. Trend prediction
2. Financial forecasting
3. Scientific modeling
4. Basic predictive analytics
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression implementation with both normal equations and gradient descent.
    
    Attributes:
        coefficients (np.ndarray): Model parameters [m, b] for y = mx + b
        method (str): Method used for fitting ('normal_equations' or 'gradient_descent')
        history (dict): Training history for gradient descent
    """
    
    def __init__(self):
        """Initialize the linear regression model."""
        self.coefficients = None
        self.method = None
        self.history = {'loss': [], 'params': []}
    
    def _add_bias(self, X):
        """
        Add a column of 1's to the input matrix for the bias term.
        
        Parameters:
            X (np.ndarray): Input features of shape (n_samples,)
        
        Returns:
            np.ndarray: Design matrix of shape (n_samples, 2)
        """
        return np.column_stack([X, np.ones_like(X)])
    
    def fit_normal_equations(self, X, y):
        """
        Fit the model using the normal equations method.
        
        Mathematical derivation:
        1. y = Xw where X is the design matrix and w = [m, b]
        2. Minimize ||y - Xw||²
        3. Solution: w = (X^T X)^(-1) X^T y
        
        Parameters:
            X (np.ndarray): Input features of shape (n_samples,)
            y (np.ndarray): Target values of shape (n_samples,)
        """
        X_design = self._add_bias(X)
        # Normal equations solution: (X^T X)^(-1) X^T y
        self.coefficients = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
        self.method = 'normal_equations'
    
    def fit_gradient_descent(self, X, y, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
        """
        Fit the model using gradient descent.
        
        Parameters:
            X (np.ndarray): Input features of shape (n_samples,)
            y (np.ndarray): Target values of shape (n_samples,)
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Maximum number of iterations
            tolerance (float): Convergence criterion for change in loss
        """
        n_samples = len(X)
        self.coefficients = np.zeros(2)  # [m, b]
        self.method = 'gradient_descent'
        
        for i in range(n_iterations):
            # Current predictions
            y_pred = self.coefficients[0] * X + self.coefficients[1]
            
            # Compute gradients
            error = y_pred - y
            grad_m = (2/n_samples) * np.sum(error * X)
            grad_b = (2/n_samples) * np.sum(error)
            
            # Update parameters
            self.coefficients[0] -= learning_rate * grad_m
            self.coefficients[1] -= learning_rate * grad_b
            
            # Compute loss
            loss = np.mean(error ** 2)
            self.history['loss'].append(loss)
            self.history['params'].append(self.coefficients.copy())
            
            # Check convergence
            if i > 0 and abs(self.history['loss'][-2] - loss) < tolerance:
                break
    
    def predict(self, X):
        """
        Make predictions for new data.
        
        Parameters:
            X (np.ndarray): Input features
        
        Returns:
            np.ndarray: Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.coefficients[0] * X + self.coefficients[1]
    
    def score(self, X, y):
        """
        Compute R² score (coefficient of determination).
        
        R² = 1 - SSres/SStot
        where:
        SSres = ∑(y - y_pred)²
        SStot = ∑(y - y_mean)²
        
        Parameters:
            X (np.ndarray): Input features
            y (np.ndarray): True target values
        
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

def visualize_fit(X, y, model, title="Linear Regression Fit"):
    """
    Visualize the regression line and data points.
    
    Parameters:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        model (LinearRegression): Fitted model
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
    
    # Plot regression line
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', label=f'Fitted line (y = {model.coefficients[0]:.2f}x + {model.coefficients[1]:.2f})')
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def test_linear_regression():
    """
    Test function to demonstrate linear regression with different scenarios.
    
    Examples demonstrate:
    1. Perfect linear relationship
    2. Noisy linear relationship
    3. Comparison of normal equations vs gradient descent
    """
    # Example 1: Perfect linear relationship
    print("\n=== Example 1: Perfect Linear Relationship ===")
    X1 = np.array([1, 2, 3, 4, 5])
    y1 = 2 * X1 + 1  # y = 2x + 1
    
    model1 = LinearRegression()
    model1.fit_normal_equations(X1, y1)
    
    print("True coefficients: m=2, b=1")
    print(f"Fitted coefficients: m={model1.coefficients[0]:.4f}, b={model1.coefficients[1]:.4f}")
    print(f"R² score: {model1.score(X1, y1):.4f}")
    
    visualize_fit(X1, y1, model1, "Perfect Linear Relationship")

    # Example 2: Noisy linear relationship
    print("\n=== Example 2: Noisy Linear Relationship ===")
    np.random.seed(42)
    X2 = np.linspace(0, 10, 100)
    y2 = 3 * X2 + 2 + np.random.normal(0, 1.5, 100)
    
    # Fit with normal equations
    model2a = LinearRegression()
    model2a.fit_normal_equations(X2, y2)
    
    # Fit with gradient descent
    model2b = LinearRegression()
    model2b.fit_gradient_descent(X2, y2)
    
    print("\nNormal Equations:")
    print(f"Fitted coefficients: m={model2a.coefficients[0]:.4f}, b={model2a.coefficients[1]:.4f}")
    print(f"R² score: {model2a.score(X2, y2):.4f}")
    
    print("\nGradient Descent:")
    print(f"Fitted coefficients: m={model2b.coefficients[0]:.4f}, b={model2b.coefficients[1]:.4f}")
    print(f"R² score: {model2b.score(X2, y2):.4f}")
    
    visualize_fit(X2, y2, model2a, "Noisy Linear Relationship (Normal Equations)")
    visualize_fit(X2, y2, model2b, "Noisy Linear Relationship (Gradient Descent)")
    
    # Plot gradient descent convergence
    plt.figure(figsize=(10, 6))
    plt.plot(model2b.history['loss'])
    plt.title('Gradient Descent Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    test_linear_regression() 