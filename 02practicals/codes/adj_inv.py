import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class AdjointInverseSolver:
    def __init__(self, n_points=101):
        """
        Initialize the solver for the inverse problem:
        -b*u'' + c*u' = f(x) with u(0) = u(1) = 0
        
        Parameters:
        n_points: Number of grid points (including boundaries)
        """
        self.n = n_points
        self.x = np.linspace(0, 1, n_points)
        self.h = 1.0 / (n_points - 1)
        
        # Interior points (excluding boundaries where u = 0)
        self.n_interior = n_points - 2
        self.x_interior = self.x[1:-1]
        
    def create_differential_matrices(self):
        """Create finite difference matrices for second and first derivatives"""
        h = self.h
        n = self.n_interior
        
        # Second derivative matrix (centered difference)
        # u''_i ≈ (u_{i-1} - 2*u_i + u_{i+1}) / h²
        diag_main = -2 * np.ones(n) / (h**2)
        diag_off = np.ones(n-1) / (h**2)
        self.D2 = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n))
        
        # First derivative matrix (centered difference)
        # u'_i ≈ (u_{i+1} - u_{i-1}) / (2*h)
        diag_left = -np.ones(n-1) / (2*h)
        diag_right = np.ones(n-1) / (2*h)
        self.D1 = diags([diag_left, diag_right], [-1, 1], shape=(n, n))
        
    def solve_forward(self, b, c, f_vals):
        """
        Solve the forward problem: -b*u'' + c*u' = f
        
        Parameters:
        b, c: coefficients
        f_vals: forcing term values at interior points
        
        Returns:
        u_interior: solution at interior points
        """
        # Create system matrix: -b*D2 + c*D1
        A = -b * self.D2 + c * self.D1
        
        # Solve Au = f
        u_interior = spsolve(A.tocsc(), f_vals)
        return u_interior
    
    def solve_adjoint(self, b, c, residual):
        """
        Solve the adjoint problem: -b*λ'' - c*λ' = 2*residual
        
        Parameters:
        b, c: coefficients
        residual: u - u_obs at interior points
        
        Returns:
        lambda_interior: adjoint solution at interior points
        """
        # Adjoint system matrix: -b*D2 - c*D1
        A_adj = -b * self.D2 - c * self.D1
        
        # Right-hand side: 2*residual
        rhs = 2 * residual
        
        # Solve adjoint system
        lambda_interior = spsolve(A_adj.tocsc(), rhs)
        return lambda_interior
    
    def compute_derivatives(self, u_interior):
        """
        Compute first and second derivatives of u at interior points
        """
        u_second_deriv = self.D2.dot(u_interior)
        u_first_deriv = self.D1.dot(u_interior)
        return u_first_deriv, u_second_deriv
    
    def compute_gradient(self, b, c, u_interior, u_obs_interior):
        """
        Compute gradient of cost function using adjoint method
        """
        # Compute residual
        residual = u_interior - u_obs_interior
        
        # Solve adjoint equation
        lambda_interior = self.solve_adjoint(b, c, residual)
        
        # Compute derivatives of u
        u_first_deriv, u_second_deriv = self.compute_derivatives(u_interior)
        
        # Compute gradients using trapezoidal integration
        h = self.h
        grad_b = h * np.sum(lambda_interior * u_second_deriv)
        grad_c = -h * np.sum(lambda_interior * u_first_deriv)
        
        return np.array([grad_b, grad_c])
    
    def cost_function(self, params, f_vals, u_obs_interior):
        """
        Evaluate cost function and its gradient
        
        Parameters:
        params: [b, c]
        f_vals: forcing term values
        u_obs_interior: observed values at interior points
        
        Returns:
        cost: value of cost function
        gradient: gradient with respect to [b, c]
        """
        b, c = params
        
        # Ensure b > 0 for ellipticity
        if b <= 0:
            return 1e10, np.array([1e6, 0])
        
        try:
            # Solve forward problem
            u_interior = self.solve_forward(b, c, f_vals)
            
            # Compute cost function (trapezoidal rule)
            residual = u_interior - u_obs_interior
            cost = 0.5 * self.h * np.sum(residual**2)
            
            # Compute gradient using adjoint method
            gradient = self.compute_gradient(b, c, u_interior, u_obs_interior)
            
            return cost, gradient
            
        except Exception as e:
            print(f"Error in cost function evaluation: {e}")
            return 1e10, np.array([1e6, 1e6])
    
    def solve_inverse_problem(self, f_vals, u_obs_interior, initial_guess=[1.0, 0.1]):
        """
        Solve the inverse problem using L-BFGS-B optimization
        
        Parameters:
        f_vals: forcing term values at interior points
        u_obs_interior: observed solution values at interior points
        initial_guess: initial guess for [b, c]
        
        Returns:
        result: optimization result
        """
        self.create_differential_matrices()
        
        # Define objective function for optimizer
        def objective(params):
            cost, grad = self.cost_function(params, f_vals, u_obs_interior)
            return cost
        
        def gradient(params):
            cost, grad = self.cost_function(params, f_vals, u_obs_interior)
            return grad
        
        # Solve using L-BFGS-B (quasi-Newton method)
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            jac=gradient,
            bounds=[(1e-6, 10.0), (-5.0, 5.0)],  # b > 0 for ellipticity
            options={'disp': True, 'maxiter': 1000, 'ftol': 1e-12}
        )
        
        return result

# Example usage and testing
def create_synthetic_data():
    """Create synthetic observed data for testing"""
    # True parameters
    b_true = 2.0
    c_true = 0.5
    
    # Create solver instance
    solver = AdjointInverseSolver(n_points=101)
    solver.create_differential_matrices()
    
    # Define forcing term f(x) = sin(2πx)
    x_interior = solver.x_interior
    f_vals = np.sin(2 * np.pi * x_interior)
    
    # Solve with true parameters to get "observed" data
    u_obs_interior = solver.solve_forward(b_true, c_true, f_vals)
    
    # Add small amount of noise
    np.random.seed(42)
    noise_level = 0.01
    u_obs_interior += noise_level * np.random.normal(0, 1, len(u_obs_interior))
    
    return f_vals, u_obs_interior, b_true, c_true, solver

def main():
    """Main function to demonstrate the inverse problem solver"""
    print("Solving Inverse Problem using Adjoint Method")
    print("=" * 50)
    
    # Create synthetic data
    f_vals, u_obs_interior, b_true, c_true, solver = create_synthetic_data()
    
    print(f"True parameters: b = {b_true:.3f}, c = {c_true:.3f}")
    
    # Solve inverse problem
    initial_guess = [1.5, 0.2]  # Initial guess (different from true values)
    print(f"Initial guess: b = {initial_guess[0]:.3f}, c = {initial_guess[1]:.3f}")
    
    result = solver.solve_inverse_problem(f_vals, u_obs_interior, initial_guess)
    
    # Display results
    b_opt, c_opt = result.x
    print(f"\nOptimization Results:")
    print(f"Estimated parameters: b = {b_opt:.6f}, c = {c_opt:.6f}")
    print(f"True parameters:      b = {b_true:.6f}, c = {c_true:.6f}")
    print(f"Errors: Δb = {abs(b_opt - b_true):.6f}, Δc = {abs(c_opt - c_true):.6f}")
    print(f"Final cost: {result.fun:.2e}")
    print(f"Optimization successful: {result.success}")
    print(f"Number of iterations: {result.nit}")
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    # Plot solutions
    plt.subplot(1, 2, 1)
    x_interior = solver.x_interior
    u_true = solver.solve_forward(b_true, c_true, f_vals)
    u_estimated = solver.solve_forward(b_opt, c_opt, f_vals)
    
    plt.plot(x_interior, u_obs_interior, 'ko', label='Observed data', markersize=3)
    plt.plot(x_interior, u_true, 'b-', label='True solution', linewidth=2)
    plt.plot(x_interior, u_estimated, 'r--', label='Estimated solution', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot forcing term
    plt.subplot(1, 2, 2)
    plt.plot(x_interior, f_vals, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Forcing Term f(x) = sin(2πx)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()