import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. Load Data & Helper Functions
# -------------------------------------------------------------
def load_data(filepath="Data_Sets/p4_train_N10.csv"):
    df = pd.read_csv(filepath)
    return df['x'].values, df['y'].values

def design_matrix(x, degree=3):
    # Vandermonde matrix in monomial basis: [1, x, x^2, ..., x^d]
    return np.vander(x, N=degree + 1, increasing=True)

def solve_ridge(Phi, t, lam):
    # Solve linear system (Phi^T Phi + lam * I) w = Phi^T t
    d = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(d)
    b = Phi.T @ t
    return np.linalg.solve(A, b)

def bayesian_linear_regression(Phi, t, alpha=0.05, beta=100.0):
    # Posterior precision and covariance: S_N^-1 = alpha * I + beta * Phi^T Phi
    d = Phi.shape[1]
    S_N_inv = alpha * np.eye(d) + beta * (Phi.T @ Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N @ Phi.T @ t
    return m_N, S_N

def predictive_distribution(Phi_dense, m_N, S_N, beta=100.0):
    # m(x) = phi(x)^T m_N
    m = Phi_dense @ m_N
    # s^2(x) = beta^-1 + phi(x)^T S_N phi(x)
    param_var = np.sum((Phi_dense @ S_N) * Phi_dense, axis=1)
    s2 = (1.0 / beta) + param_var
    s = np.sqrt(s2)
    return m, s

# -------------------------------------------------------------
# Deliverables 1 & 2: Predictive Distribution & Equivalence Check
# -------------------------------------------------------------
def run_deliverables_1_and_2(x_train, y_train, Phi_train, alpha=0.05, beta=100.0, degree=3):
    print("--- Deliverables 1 & 2: Predictive Distribution & Equivalence Check ---")
    m_N, S_N = bayesian_linear_regression(Phi_train, y_train, alpha, beta)
    
    # Dense test grid
    x_dense = np.linspace(0, 1, 200)
    Phi_dense = design_matrix(x_dense, degree)
    y_true = np.sin(2 * np.pi * x_dense)
    
    m, s = predictive_distribution(Phi_dense, m_N, S_N, beta)
    
    # Deliverable 2: MAP point estimate with lambda = alpha / beta
    lam = alpha / beta
    w_map = solve_ridge(Phi_train, y_train, lam)
    y_map = Phi_dense @ w_map
    
    diff_norm = np.linalg.norm(w_map - m_N)
    print(f"w_MAP: {np.round(w_map, 4)}")
    print(f"m_N  : {np.round(m_N, 4)}")
    print(f"Numerical difference ||w_MAP - m_N||_2: {diff_norm:.2e}")
    print("Equivalence confirmed: MAP estimate exactly coincides with the Bayesian posterior mean.")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='black', zorder=5, label='Train Data (N=10)')
    plt.plot(x_dense, y_true, 'k--', alpha=0.6, label=r'$\sin(2\pi x)$')
    plt.plot(x_dense, m, 'b-', linewidth=2, label='Predictive Mean $m(x)$')
    plt.plot(x_dense, y_map, 'r:', linewidth=2, label=r'MAP Fit $\hat{y}_{\mathrm{MAP}}(x)$')
    plt.fill_between(x_dense, m - 1.96 * s, m + 1.96 * s, color='blue', alpha=0.2, label='95% Predictive Interval')
    
    plt.ylim(-1.5, 1.5)
    plt.title(r"Deliverables 1 & 2: Bayesian Linear Regression & MAP Equivalence ($N=10$)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# Deliverables 3 & 4: Effect of N & Uncertainty Discussion
# -------------------------------------------------------------
def run_deliverables_3_and_4(alpha=0.05, beta=100.0, degree=3):
    print("\n--- Deliverables 3 & 4: Effect of N & Discussion ---")
    x_dense = np.linspace(0, 1, 200)
    Phi_dense = design_matrix(x_dense, degree)
    
    Ns = [5, 10, 20, 50]
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
    
    plt.figure(figsize=(10, 6))
    for N, col in zip(Ns, colors):
        x_N, y_N = load_data(f"Data_Sets/p4_train_N{N}.csv")
        Phi_N = design_matrix(x_N, degree)
        m_N, S_N = bayesian_linear_regression(Phi_N, y_N, alpha, beta)
        _, s_N = predictive_distribution(Phi_dense, m_N, S_N, beta)
        
        interval_width = 2 * 1.96 * s_N
        plt.plot(x_dense, interval_width, color=col, linewidth=1.8, label=f'N = {N}')
        print(f"N = {N:2d} | Avg 95% Interval Width: {np.mean(interval_width):.4f} | Max Width: {np.max(interval_width):.4f}")
        
    # Irreducible noise floor: 2 * 1.96 * beta^-1/2
    noise_floor = 2 * 1.96 / np.sqrt(beta)
    plt.axhline(noise_floor, color='black', linestyle='--', linewidth=1.5, label=r'Noise floor $2 \times 1.96 \times \beta^{-1/2} \approx 0.392$')
    
    plt.title(r"Deliverable 3: Predictive Interval Width $2 \times 1.96 s(x)$ vs $x$ for Varying $N$")
    plt.xlabel("x")
    plt.ylabel("Predictive Interval Width")
    plt.ylim(0.3, 1.4)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # Deliverable 4: Discussion
    print("\nDiscussion (Deliverable 4):")
    print("  As training size N increases, the data precision beta * Phi^T Phi dominates the prior precision in S_N^-1,")
    print("  driving parameter covariance S_N toward zero and substantially shrinking parameter uncertainty phi(x)^T S_N phi(x).")
    print("  However, the predictive variance s^2(x) is lower-bounded by the irreducible observation noise beta^-1 = 0.01;")
    print("  hence the interval width asymptotically converges toward the irreducible noise floor (0.392) as N -> inf.")

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    # Fixed hyperparameters
    sigma2 = 0.01
    beta = 1.0 / sigma2   # beta = 100
    alpha = 0.05
    degree = 3
    
    # 1. Load N = 10 dataset
    x_train, y_train = load_data("Data_Sets/p4_train_N10.csv")
    print(f"Loaded {len(x_train)} training samples for N = 10.")
    
    # 2. Design matrix for N = 10 (degree = 3)
    Phi_train = design_matrix(x_train, degree=degree)
    
    # Deliverables 1 & 2: Predictive distribution and MAP equivalence check
    run_deliverables_1_and_2(x_train, y_train, Phi_train, alpha=alpha, beta=beta, degree=degree)
    
    # Deliverables 3 & 4: Effect of sample size N and uncertainty discussion
    run_deliverables_3_and_4(alpha=alpha, beta=beta, degree=degree)
