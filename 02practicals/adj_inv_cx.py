import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class FunctionalAdjointSolver:
    def __init__(self, n_points=101):
        """
        Initialize the solver for the inverse problem:
        -u'' + c(x)*u' = f(x) with u(0) = u(1) = 0
        where c(x) is unknown function to be estimated
        
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
        
    def create_c_matrix(self, c_vals):
        """
        Create diagonal matrix with c(x) values for c(x)*u' term
        
        Parameters:
        c_vals: values of c(x) at interior points
        
        Returns:
        C_D1: matrix representing c(x)*D1
        """
        C = diags(c_vals, 0, shape=(self.n_interior, self.n_interior))
        return C.dot(self.D1)
        
    def solve_forward(self, c_vals, f_vals):
        """
        Solve the forward problem: -u'' + c(x)*u' = f
        
        Parameters:
        c_vals: values of c(x) at interior points
        f_vals: forcing term values at interior points
        
        Returns:
        u_interior: solution at interior points
        """
        # Create system matrix: -D2 + c(x)*D1
        C_D1 = self.create_c_matrix(c_vals)
        A = -self.D2 + C_D1
        
        # Solve Au = f
        u_interior = spsolve(A.tocsc(), f_vals)
        return u_interior
    
    def solve_adjoint(self, c_vals, residual):
        """
        Solve the adjoint problem: -λ'' - c(x)*λ' = 2*residual
        
        Parameters:
        c_vals: values of c(x) at interior points
        residual: u - u_obs at interior points
        
        Returns:
        lambda_interior: adjoint solution at interior points
        """
        # Adjoint system matrix: -D2 - c(x)*D1
        C_D1 = self.create_c_matrix(c_vals)
        A_adj = -self.D2 - C_D1
        
        # Right-hand side: 2*residual
        rhs = 2 * residual
        
        # Solve adjoint system
        lambda_interior = spsolve(A_adj.tocsc(), rhs)
        return lambda_interior
    
    def compute_u_derivative(self, u_interior):
        """
        Compute first derivative of u at interior points
        """
        u_first_deriv = self.D1.dot(u_interior)
        return u_first_deriv
    
    def compute_functional_gradient(self, c_vals, u_interior, u_obs_interior):
        """
        Compute functional gradient δJ/δc(x) using adjoint method
        
        Returns:
        grad_c: gradient with respect to c(x) at each interior point
        """
        # Compute residual
        residual = u_interior - u_obs_interior
        
        # Solve adjoint equation
        lambda_interior = self.solve_adjoint(c_vals, residual)
        
        # Compute first derivative of u
        u_first_deriv = self.compute_u_derivative(u_interior)
        
        # Functional gradient: δJ/δc(x) = -λ(x) * u'(x)
        grad_c = -lambda_interior * u_first_deriv
        
        return grad_c
    
    def cost_function(self, c_vals, f_vals, u_obs_interior):
        """
        Evaluate cost function and its functional gradient
        
        Parameters:
        c_vals: values of c(x) at interior points
        f_vals: forcing term values
        u_obs_interior: observed values at interior points
        
        Returns:
        cost: value of cost function
        gradient: functional gradient with respect to c(x)
        """
        try:
            # Solve forward problem
            u_interior = self.solve_forward(c_vals, f_vals)
            
            # Compute cost function (trapezoidal rule)
            residual = u_interior - u_obs_interior
            cost = 0.5 * self.h * np.sum(residual**2)
            
            # Compute functional gradient using adjoint method
            gradient = self.compute_functional_gradient(c_vals, u_interior, u_obs_interior)
            
            return cost, gradient
            
        except Exception as e:
            print(f"Error in cost function evaluation: {e}")
            return 1e10, np.ones_like(c_vals) * 1e6
    
    def solve_inverse_problem(self, f_vals, u_obs_interior, initial_guess=None, 
                            bounds=None, regularization=0.0):
        """
        Solve the inverse problem using L-BFGS-B optimization
        
        Parameters:
        f_vals: forcing term values at interior points
        u_obs_interior: observed solution values at interior points
        initial_guess: initial guess for c(x) values (default: zeros)
        bounds: bounds for c(x) values (default: [-5, 5])
        regularization: L2 regularization parameter
        
        Returns:
        result: optimization result
        """
        self.create_differential_matrices()
        
        # Set default initial guess
        if initial_guess is None:
            initial_guess = np.zeros(self.n_interior)
            
        # Set default bounds
        if bounds is None:
            bounds = [(-5.0, 5.0)] * self.n_interior
        
        # Store regularization parameter
        self.regularization = regularization
        
        # Define objective function for optimizer
        def objective(c_vals):
            cost, grad = self.cost_function(c_vals, f_vals, u_obs_interior)
            
            # Add L2 regularization
            if self.regularization > 0:
                reg_cost = 0.5 * self.regularization * self.h * np.sum(c_vals**2)
                cost += reg_cost
                
            return cost
        
        def gradient(c_vals):
            cost, grad = self.cost_function(c_vals, f_vals, u_obs_interior)
            
            # Add regularization gradient
            if self.regularization > 0:
                grad += self.regularization * self.h * c_vals
                
            return grad
        
        # Solve using L-BFGS-B (quasi-Newton method)
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'disp': True, 'maxiter': 1000, 'ftol': 1e-12}
        )
        
        return result

# Example usage and testing
def create_synthetic_data():
    """Create synthetic observed data for testing"""
    # True spatially-varying coefficient c(x)
    def c_true_func(x):
        return 0.5 + 0.3 * np.sin(3 * np.pi * x) + 0.2 * np.cos(5 * np.pi * x)
    
    # Create solver instance
    solver = FunctionalAdjointSolver(n_points=101)
    solver.create_differential_matrices()
    
    # Define true c(x) at interior points
    x_interior = solver.x_interior
    c_true_vals = c_true_func(x_interior)
    
    # Define forcing term f(x) = sin(2πx)
    f_vals = np.sin(2 * np.pi * x_interior)
    
    # Solve with true parameters to get "observed" data
    u_obs_interior = solver.solve_forward(c_true_vals, f_vals)
    
    # Add small amount of noise
    np.random.seed(42)
    noise_level = 0.005
    u_obs_interior += noise_level * np.random.normal(0, 1, len(u_obs_interior))
    
    return f_vals, u_obs_interior, c_true_vals, c_true_func, solver

def main():
    """Main function to demonstrate the functional inverse problem solver"""
    print("Solving Functional Inverse Problem using Adjoint Method")
    print("Estimating spatially-varying coefficient c(x)")
    print("=" * 60)
    
    # Create synthetic data
    f_vals, u_obs_interior, c_true_vals, c_true_func, solver = create_synthetic_data()
    
    print(f"Problem setup:")
    print(f"- Grid points: {solver.n}")
    print(f"- Interior points: {solver.n_interior}")
    print(f"- Unknown function: c(x)")
    
    # Solve inverse problem with regularization
    initial_guess = np.ones(solver.n_interior) * 0.2  # Constant initial guess
    regularization = 1e-4  # Small regularization for smoothness
    
    print(f"- Initial guess: c(x) = {initial_guess[0]:.3f} (constant)")
    print(f"- Regularization parameter: {regularization:.1e}")
    
    result = solver.solve_inverse_problem(
        f_vals, u_obs_interior, 
        initial_guess=initial_guess,
        regularization=regularization
    )
    
    # Extract results
    c_estimated_vals = result.x
    
    # Display results
    print(f"\nOptimization Results:")
    print(f"Final cost: {result.fun:.2e}")
    print(f"Optimization successful: {result.success}")
    print(f"Number of iterations: {result.nit}")
    
    # Compute errors
    c_error_l2 = np.sqrt(solver.h * np.sum((c_estimated_vals - c_true_vals)**2))
    c_error_max = np.max(np.abs(c_estimated_vals - c_true_vals))
    
    print(f"L2 error in c(x): {c_error_l2:.6f}")
    print(f"Max error in c(x): {c_error_max:.6f}")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot estimated vs true c(x)
    x_interior = solver.x_interior
    axes[0,0].plot(x_interior, c_true_vals, 'b-', label='True c(x)', linewidth=3)
    axes[0,0].plot(x_interior, c_estimated_vals, 'r--', label='Estimated c(x)', linewidth=2)
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('c(x)')
    axes[0,0].set_title('Coefficient Function Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot error in c(x)
    error_c = c_estimated_vals - c_true_vals
    axes[0,1].plot(x_interior, error_c, 'g-', linewidth=2)
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('Error in c(x)')
    axes[0,1].set_title('Error in Estimated Coefficient')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot solutions
    u_true = solver.solve_forward(c_true_vals, f_vals)
    u_estimated = solver.solve_forward(c_estimated_vals, f_vals)
    
    axes[1,0].plot(x_interior, u_obs_interior, 'ko', label='Observed data', markersize=3)
    axes[1,0].plot(x_interior, u_true, 'b-', label='True solution', linewidth=2)
    axes[1,0].plot(x_interior, u_estimated, 'r--', label='Estimated solution', linewidth=2)
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('u(x)')
    axes[1,0].set_title('Solution Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot forcing term
    axes[1,1].plot(x_interior, f_vals, 'g-', linewidth=2)
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('f(x)')
    axes[1,1].set_title('Forcing Term f(x) = sin(2πx)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test with different regularization parameters
    print("\nTesting different regularization parameters:")
    reg_params = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    errors = []
    
    for reg in reg_params:
        result_reg = solver.solve_inverse_problem(
            f_vals, u_obs_interior, 
            initial_guess=initial_guess,
            regularization=reg
        )
        c_reg = result_reg.x
        error_reg = np.sqrt(solver.h * np.sum((c_reg - c_true_vals)**2))
        errors.append(error_reg)
        print(f"  Regularization {reg:.1e}: L2 error = {error_reg:.6f}")
    
    # Plot regularization effect
    plt.figure(figsize=(8, 6))
    plt.semilogx(reg_params, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Regularization Parameter')
    plt.ylabel('L2 Error in c(x)')
    plt.title('Effect of Regularization on Reconstruction Accuracy')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()