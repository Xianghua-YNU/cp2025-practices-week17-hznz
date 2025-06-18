#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    Solve 2D Poisson equation using relaxation method.
    
    Args:
        M (int): Number of grid points per side
        target (float): Convergence tolerance
        max_iterations (int): Maximum number of iterations
    
    Returns:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): Electric potential distribution
            iterations (int): Number of iterations performed
            converged (bool): Whether solution converged
    """
    # Grid spacing
    h = 1.0
    
    # Initialize potential array with boundary conditions
    phi = np.zeros((M+1, M+1), dtype=float)
    phi_prev = np.copy(phi)
    
    # Set up charge density distribution
    rho = np.zeros((M+1, M+1), dtype=float)
    
    # Scale charge positions based on grid size
    # For M=100: pos (60:80, 20:40), neg (20:40, 60:80)
    pos_y1, pos_y2 = int(0.6*M), int(0.8*M)
    pos_x1, pos_x2 = int(0.2*M), int(0.4*M)
    neg_y1, neg_y2 = int(0.2*M), int(0.4*M)
    neg_x1, neg_x2 = int(0.6*M), int(0.8*M)
    
    rho[pos_y1:pos_y2, pos_x1:pos_x2] = 1.0   # Positive charge
    rho[neg_y1:neg_y2, neg_x1:neg_x2] = -1.0  # Negative charge
    
    # Relaxation iteration
    delta = 1.0
    iterations = 0
    converged = False
    
    while delta > target and iterations < max_iterations:
        # Update interior points using finite difference formula
        phi[1:-1, 1:-1] = 0.25 * (phi[0:-2, 1:-1] + phi[2:, 1:-1] + 
                                   phi[1:-1, :-2] + phi[1:-1, 2:] + 
                                   h*h * rho[1:-1, 1:-1])
        
        # Calculate maximum change for convergence check
        delta = np.max(np.abs(phi - phi_prev))
        
        # Update previous solution
        phi_prev = np.copy(phi)
        iterations += 1
    
    converged = bool(delta <= target)
    
    return phi, iterations, converged

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """
    Visualize the electric potential distribution.
    
    Args:
        phi (np.ndarray): Electric potential array
        M (int): Grid size
    """
    plt.figure(figsize=(10, 8))
    
    # Create potential plot
    im = plt.imshow(phi, extent=[0, M, 0, M], origin='lower', 
                    cmap='RdBu_r', interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Electric Potential (V)', fontsize=12)
    
    # Mark charge locations
    plt.fill_between([20, 40], [60, 60], [80, 80], alpha=0.3, color='red', label='Positive Charge')
    plt.fill_between([60, 80], [20, 20], [40, 40], alpha=0.3, color='blue', label='Negative Charge')
    
    # Add labels and title
    plt.xlabel('x (grid points)', fontsize=12)
    plt.ylabel('y (grid points)', fontsize=12)
    plt.title('Electric Potential Distribution\nPoisson Equation with Positive and Negative Charges', fontsize=14)
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    Analyze and print solution statistics.
    
    Args:
        phi (np.ndarray): Electric potential array
        iterations (int): Number of iterations
        converged (bool): Convergence status
    """
    print(f"Solution Analysis:")
    print(f"  Iterations: {iterations}")
    print(f"  Converged: {converged}")
    print(f"  Max potential: {np.max(phi):.6f} V")
    print(f"  Min potential: {np.min(phi):.6f} V")
    print(f"  Potential range: {np.max(phi) - np.min(phi):.6f} V")
    
    # Find locations of extrema
    max_idx = np.unravel_index(np.argmax(phi), phi.shape)
    min_idx = np.unravel_index(np.argmin(phi), phi.shape)
    print(f"  Max potential location: ({max_idx[0]}, {max_idx[1]})")
    print(f"  Min potential location: ({min_idx[0]}, {min_idx[1]})")

if __name__ == "__main__":
    # Solve the Poisson equation
    print("Solving 2D Poisson equation with relaxation method...")
    
    # Parameters
    M = 100
    target = 1e-6
    max_iter = 10000
    
    # Solve
    phi, iterations, converged = solve_poisson_equation(M, target, max_iter)
    
    # Analyze results
    analyze_solution(phi, iterations, converged)
    
    # Visualize
    visualize_solution(phi, M)
    
    # Additional analysis: potential along center lines
    plt.figure(figsize=(12, 5))
    
    # Horizontal cross-section
    plt.subplot(1, 2, 1)
    center_y = M // 2
    plt.plot(phi[center_y, :], 'b-', linewidth=2)
    plt.xlabel('x (grid points)')
    plt.ylabel('Potential (V)')
    plt.title(f'Potential along y = {center_y}')
    plt.grid(True, alpha=0.3)
    
    # Vertical cross-section
    plt.subplot(1, 2, 2)
    center_x = M // 2
    plt.plot(phi[:, center_x], 'r-', linewidth=2)
    plt.xlabel('y (grid points)')
    plt.ylabel('Potential (V)')
    plt.title(f'Potential along x = {center_x}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
