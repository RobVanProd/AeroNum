#!/usr/bin/env python3
"""
NumPy Computational Benchmark - Identical operations to AeroNum
This performs the same mathematical computations for direct comparison
"""

import numpy as np
import time
import sys

def main():
    start_time = time.perf_counter()
    
    # Matrix computation - 3x3 matrix operations
    # Matrix A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix_a = np.array([
        [1, 2, 3],
        [4, 5, 6], 
        [7, 8, 9]
    ])
    
    # Matrix B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    matrix_b = np.array([
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ])
    
    # Vector operations - 3D vectors
    vector_u = np.array([1, 2, 3])
    vector_v = np.array([4, 5, 6])
    
    # Scalar values for computations
    scalar_alpha = 2
    scalar_beta = 3
    
    # Array elements for numerical operations
    array_data = np.array([10, 20, 30, 40, 50])
    
    # Polynomial coefficients for evaluation
    # p(x) = 2x^3 + 3x^2 + 4x + 5
    poly_coeffs = np.array([2, 3, 4, 5])  # [x^3, x^2, x^1, x^0]
    poly_x = 2
    
    # Statistical data points
    data_points = np.array([15, 25, 35, 45, 55])
    
    # Trigonometric operations (for x = 0.5)
    x_trig = 0.5
    sin_result = np.sin(x_trig)
    cos_result = np.cos(x_trig)
    tan_result = np.tan(x_trig)
    
    # ACTUAL COMPUTATIONAL OPERATIONS
    
    # 1. Matrix multiplication C = A * B
    matrix_mult_result = np.dot(matrix_a, matrix_b)
    c11 = matrix_mult_result[0, 0]  # Should be 30
    
    # 2. Dot product u · v
    dot_product_result = np.dot(vector_u, vector_v)  # Should be 32
    
    # 3. Vector magnitude squared |u|²
    vector_magnitude_sq = np.dot(vector_u, vector_u)  # Should be 14
    
    # 4. Polynomial evaluation p(2) = 2*8 + 3*4 + 4*2 + 5
    polynomial_result = np.polyval(poly_coeffs, poly_x)  # Should be 41
    
    # 5. Statistical mean
    statistical_mean = np.mean(data_points)  # Should be 35
    
    # 6. Matrix determinant
    matrix_det = np.linalg.det(matrix_a)  # Should be 0 (singular matrix)
    
    # 7. Vector cross product (first component)
    cross_product = np.cross(vector_u, vector_v)
    cross_first = cross_product[0]  # Should be -3
    
    # 8. Matrix trace (sum of diagonal elements)
    matrix_trace = np.trace(matrix_a)  # Should be 15
    
    # 9. Array sum and statistics
    array_sum = np.sum(array_data)  # Should be 150
    array_std = np.std(data_points)  # Standard deviation
    
    # 10. Numerical integration approximation (trapezoidal rule)
    x_vals = np.linspace(0, 1, 100)
    y_vals = x_vals ** 2
    integration_result = np.trapz(y_vals, x_vals)  # Should be ≈ 0.333
    
    # 11. Linear algebra operations
    # Matrix inverse (if possible)
    try:
        matrix_inv = np.linalg.inv(matrix_b)
        inv_trace = np.trace(matrix_inv)
    except:
        inv_trace = 0
    
    # 12. Eigenvalue computation (largest eigenvalue)
    eigenvals = np.linalg.eigvals(matrix_a)
    max_eigenval = np.max(np.real(eigenvals))
    
    # Final computational result combining all operations
    # This represents a complex numerical computation result
    final_computation = int(c11)  # Use matrix multiplication result (30)
    
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1_000_000  # Convert to microseconds
    
    # Output results for verification
    print(f"NumPy Computational Results:")
    print(f"Matrix multiplication C[0,0]: {c11}")
    print(f"Dot product u·v: {dot_product_result}")
    print(f"Vector magnitude squared: {vector_magnitude_sq}")
    print(f"Polynomial p(2): {polynomial_result}")
    print(f"Statistical mean: {statistical_mean}")
    print(f"Matrix determinant: {matrix_det:.6f}")
    print(f"Cross product first component: {cross_first}")
    print(f"Matrix trace: {matrix_trace}")
    print(f"Array sum: {array_sum}")
    print(f"Integration result: {integration_result:.6f}")
    print(f"Max eigenvalue: {max_eigenval:.6f}")
    print(f"Final computation result: {final_computation}")
    print(f"NumPy execution time: {execution_time:.1f} microseconds")
    
    return final_computation

if __name__ == "__main__":
    result = main()
    sys.exit(result)

