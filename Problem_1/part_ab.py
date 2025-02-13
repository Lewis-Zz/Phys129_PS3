import numpy as np
from scipy.integrate import nquad

def compute_numeric_integral(A, w):

    # Convert inputs to numpy arrays and validate

    A = np.array(A, dtype=np.float64)
    w = np.array(w, dtype=np.float64).flatten()
    N = len(w)
    
    # Validate matrix dimensions
    if A.shape != (N, N):
        raise ValueError(f"Matrix A must be {N}x{N} for vector w of length {N}")
        
    # Define the integrand function
    def integrand(*args):
        v = np.array(args)
        quadratic_term = -0.5 * np.sum(v @ A @ v)  # -½ vᵀAv
        linear_term = np.dot(v, w)                # vᵀw
        return np.exp(quadratic_term + linear_term)
    
    # Set up infinite integration limits for all dimensions
    limits = [(-np.inf, np.inf) for _ in range(N)]
    
    # Compute the numerical integral
    result, _ = nquad(integrand, limits)
    return result

def compute_closed_form(A, w):

    # Convert inputs to numpy arrays and validate
    A = np.array(A, dtype=np.float64)
    w = np.array(w, dtype=np.float64).flatten()
    N = len(w)
    
    # Validate matrix dimensions
    if A.shape != (N, N):
        raise ValueError(f"Matrix A must be {N}x{N} for vector w of length {N}")
    
    try:
        # Compute inverse of A
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix A is singular and cannot be inverted")
    
    # Compute determinant of A
    det_A = np.linalg.det(A)
    
    # Compute normalization term: √((2π)^N / det(A))
    normalization = np.sqrt(((2 * np.pi) ** N) / det_A)
    
    # Compute quadratic form: ½ wᵀ A⁻¹ w
    quadratic_form = 0.5 * w.T @ A_inv @ w
    
    # Compute the closed-form expression
    closed_form = normalization * np.exp(quadratic_form)
    return closed_form

if __name__ == "__main__":
    # Define A, A', and w as given in the question
    A = [
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6]
    ]
    A_prime = [
        [4, 2, 1],
        [2, 1, 3],
        [1, 3, 6]
    ]
    w = [1, 2, 3]

    # First test: matrix A
    I_num_A = compute_numeric_integral(A, w)
    I_ana_A = compute_closed_form(A, w)
    print("=== Testing matrix A ===")
    print(f"  Numerical integral:  {I_num_A}")
    print(f"  Closed-form result:  {I_ana_A}\n")

    # Second test: matrix A'
    I_num_Ap = compute_numeric_integral(A_prime, w)
    I_ana_Ap = compute_closed_form(A_prime, w)
    print("=== Testing matrix A' ===")
    print(f"  Numerical integral:  {I_num_Ap}")
    print(f"  Closed-form result:  {I_ana_Ap}")
## For matrixA the result nicely matches:  
## Numerical integral:  4.275823659021463
## Closed-form result:  4.275823659011516
## However, for matrix A', the integral diverge
