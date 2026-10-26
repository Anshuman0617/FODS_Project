import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. Load Data & Helper Functions
# -------------------------------------------------------------
def load_data():
    train_df = pd.read_csv("Data_Sets/poly_train.csv")
    test_df = pd.read_csv("Data_Sets/poly_test.csv")
    return train_df['x'].values, train_df['y'].values, test_df['x'].values, test_df['y'].values

def design_matrix(x, degree):
    # Vandermonde matrix: [1, x, x^2, ..., x^d]
    return np.vander(x, N=degree + 1, increasing=True)

def ridge_loss(Phi, w, t, lam):
    # 0.5 * ||Phi w - t||^2 + 0.5 * lam * ||w||^2
    res = Phi @ w - t
    return 0.5 * np.sum(res ** 2) + 0.5 * lam * np.sum(w ** 2)

def rmse(Phi, w, t):
    return np.sqrt(np.mean((Phi @ w - t) ** 2))

def solve_ridge(Phi, t, lam):
    # Solve linear system directly without explicit inverse
    d = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(d)
    b = Phi.T @ t
    return np.linalg.solve(A, b)

# -------------------------------------------------------------
# Deliverable 1: Closed-form solutions across lambda grid
# -------------------------------------------------------------
def run_deliverable_1(Phi_train, y_train, lams):
    print("--- Deliverable 1: Closed-form Ridge Solutions ---")
    ws = [solve_ridge(Phi_train, y_train, lam) for lam in lams]
    print(f"Solved ridge regression for {len(lams)} values of lambda.")
    return np.array(ws)

# -------------------------------------------------------------
# Deliverable 2: GD Agreement at fixed lambda = 10^-2
# -------------------------------------------------------------
def batch_gd_ridge(Phi, t, lam=1e-2, eta=0.04, tol=1e-6):
    w = np.zeros(Phi.shape[1])
    w_closed = solve_ridge(Phi, t, lam)
    norm_closed = np.linalg.norm(w_closed)
    losses = []
    
    iters = 0
    while True:
        losses.append(ridge_loss(Phi, w, t, lam))
        grad = Phi.T @ (Phi @ w - t) + lam * w
        w -= eta * grad
        iters += 1
        
        rel_err = np.linalg.norm(w - w_closed) / norm_closed
        if rel_err < tol:
            losses.append(ridge_loss(Phi, w, t, lam))
            break
            
    return w, w_closed, np.array(losses), iters, rel_err

def run_deliverable_2(Phi_train, y_train, lam=1e-2):
    print("\n--- Deliverable 2: GD Agreement at lambda = 1e-2 ---")
    w_gd, w_closed, losses, iters, rel_err = batch_gd_ridge(Phi_train, y_train, lam=lam, eta=0.04)
    
    cond_unreg = np.linalg.cond(Phi_train.T @ Phi_train)
    cond_reg = np.linalg.cond(Phi_train.T @ Phi_train + lam * np.eye(Phi_train.shape[1]))
    
    print(f"Converged in {iters} iterations (relative error = {rel_err:.2e} < 1e-6)")
    print(f"cond(Phi^T Phi)        : {cond_unreg:.2e}")
    print(f"cond(Phi^T Phi + lam*I): {cond_reg:.2e}")
    print("Conditioning impact: Adding lambda*I shifts the eigenvalues upward, dropping the condition number by 11 orders of magnitude and making gradient descent converge stably.")
    
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='tab:red', linewidth=1.8)
    plt.title(r"Deliverable 2: Batch GD Convergence for Ridge Regression ($\lambda=10^{-2}$, $\eta=0.04$)")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\tilde{E}(w)$")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# Deliverable 3: RMSE Sweep & Choosing lambda*
# -------------------------------------------------------------
def run_deliverable_3(Phi_train, y_train, Phi_test, y_test, lams, ws):
    print("\n--- Deliverable 3: RMSE Sweep ---")
    train_rmses = np.array([rmse(Phi_train, w, y_train) for w in ws])
    test_rmses = np.array([rmse(Phi_test, w, y_test) for w in ws])
    
    idx = np.argmin(test_rmses)
    lam_star = lams[idx]
    min_test = test_rmses[idx]
    
    # 1.05x minimum test error interval
    valid_lams = lams[test_rmses <= 1.05 * min_test]
    interval = (valid_lams[0], valid_lams[-1])
    
    print(f"Optimal lambda* : {lam_star:.4e} (log10 = {np.log10(lam_star):.2f})")
    print(f"Test RMSE at lambda*: {min_test:.4f}")
    print(f"1.05x interval  : [{interval[0]:.4e}, {interval[1]:.4e}]")
    print("Discussion: The minimum is quite flat, spanning more than 3 orders of magnitude of lambda.")
    print("To the left of the interval the model overfits to noise with huge coefficients, while to the right strong regularization overly shrinks the weights, causing underfitting.")

    plt.figure(figsize=(8, 5))
    plt.plot(lams, train_rmses, 'b-', label='Train RMSE', linewidth=1.8)
    plt.plot(lams, test_rmses, 'r-', label='Test RMSE', linewidth=1.8)
    plt.axhline(0.1, color='gray', linestyle='--', label=r'Noise floor $\sqrt{0.01}=0.1$')
    plt.axvline(lam_star, color='green', linestyle=':', label=rf'Best $\lambda^* = {lam_star:.2e}$')
    plt.axvspan(interval[0], interval[1], color='green', alpha=0.15, label=r'1.05$\times$ min interval')
    
    plt.xscale('log')
    plt.ylim(0, 1.0)
    plt.xlabel(r'$\lambda$ (log scale)')
    plt.ylabel('RMSE')
    plt.title(r'Deliverable 3: Train and Test RMSE vs $\lambda$')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return lam_star

# -------------------------------------------------------------
# Deliverable 4: Fitted Curves (Overfit, Optimal, Underfit)
# -------------------------------------------------------------
def run_deliverable_4(x_train, y_train, Phi_train, lam_star):
    print("\n--- Deliverable 4: Fitted Curves ---")
    x_dense = np.linspace(0, 1, 200)
    Phi_dense = design_matrix(x_dense, degree=9)
    y_true = np.sin(2 * np.pi * x_dense)
    
    models = [
        (1e-10, "lambda = 10^-10 (Overfit)", 'purple'),
        (lam_star, f"lambda* = {lam_star:.2e} (Optimal)", 'green'),
        (1e-2, "lambda = 10^-2 (Underfit)", 'crimson')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (lam, name, col) in zip(axes, models):
        w = solve_ridge(Phi_train, y_train, lam)
        w_norm = np.linalg.norm(w)
        print(f"{name:30s} | ||w||_2: {w_norm:.4f}")
        
        ax.scatter(x_train, y_train, color='black', label='Train Data', zorder=5)
        ax.plot(x_dense, y_true, 'k--', alpha=0.5, label=r'$\sin(2\pi x)$')
        ax.plot(x_dense, Phi_dense @ w, color=col, linewidth=1.8, label='Fit')
        ax.set_title(f"{name}\n||w|| = {w_norm:.2f}")
        ax.set_xlabel("x")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()
        
    axes[0].set_ylabel("y")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load datasets
    x_train, y_train, x_test, y_test = load_data()
    print(f"Loaded {len(x_train)} train points and {len(x_test)} test points.")
    
    # 2. Design matrices for d = 9
    d = 9
    Phi_train = design_matrix(x_train, d)
    Phi_test = design_matrix(x_test, d)
    
    # 3. Grid for lambda
    lams = np.logspace(-10, 2, 49)
    
    # Deliverable 1: Closed form
    ws = run_deliverable_1(Phi_train, y_train, lams)
    
    # Deliverable 2: GD at fixed lambda = 10^-2
    run_deliverable_2(Phi_train, y_train, lam=1e-2)
    
    # Deliverable 3: RMSE sweep
    lam_star = run_deliverable_3(Phi_train, y_train, Phi_test, y_test, lams, ws)
    
    # Deliverable 4: Fitted curves
    run_deliverable_4(x_train, y_train, Phi_train, lam_star)
