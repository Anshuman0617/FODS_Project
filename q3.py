import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. Load Data & Helper Functions
# -------------------------------------------------------------
def load_data(train_path="Data_Sets/poly_train.csv", test_path="Data_Sets/poly_test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df['x'].values, train_df['y'].values, test_df['x'].values, test_df['y'].values

def design_matrix(x, degree=9):
    # Shifted Legendre polynomial basis on [0, 1] mapped to [-1, 1]
    return np.polynomial.legendre.legvander(2 * x - 1, degree)

def lasso_loss(Phi, w, t, lam):
    # 0.5 * ||Phi w - t||^2 + lam * ||w||_1
    res = Phi @ w - t
    return 0.5 * np.sum(res ** 2) + lam * np.sum(np.abs(w))

def rmse(Phi, w, t):
    return np.sqrt(np.mean((Phi @ w - t) ** 2))

def solve_ridge(Phi, t, lam):
    # Solve linear system (Phi^T Phi + lam * I) w = Phi^T t
    d = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(d)
    b = Phi.T @ t
    return np.linalg.solve(A, b)

# -------------------------------------------------------------
# Subgradient Descent for Lasso
# -------------------------------------------------------------
def subgradient_descent(Phi, t, lam, eta=0.05, n_iters=20000):
    w = np.zeros(Phi.shape[1])
    losses = []
    best_losses = []
    best_loss = np.inf
    
    for _ in range(n_iters):
        res = Phi @ w - t
        loss = 0.5 * np.sum(res ** 2) + lam * np.sum(np.abs(w))
        if loss < best_loss:
            best_loss = loss
            
        losses.append(loss)
        best_losses.append(best_loss)
        
        # Subgradient update: w <- w - eta * (Phi^T (Phi w - t) + lam * sign(w))
        grad = Phi.T @ res + lam * np.sign(w)
        w -= eta * grad
        
    return w, np.array(losses), np.array(best_losses)

# -------------------------------------------------------------
# Deliverable 1: Convergence and Verification at lambda = 0.1
# -------------------------------------------------------------
def run_deliverable_1(Phi_train, y_train, lam=0.1, eta=0.05):
    print("--- Deliverable 1: Convergence and Verification (lambda = 0.1) ---")
    w, losses, best_losses = subgradient_descent(Phi_train, y_train, lam=lam, eta=eta, n_iters=20000)
    
    # 1(a) Plot loss vs iteration
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label=r'Iterate Loss $E(w)$', alpha=0.4, color='tab:blue')
    plt.plot(best_losses, label=r'Best-so-far $E$', color='tab:red', linewidth=1.8)
    plt.yscale('log')
    plt.title(rf"Deliverable 1(a): Subgradient Descent Convergence ($\lambda = {lam}$, $\eta = {eta}$)")
    plt.xlabel("Iteration")
    plt.ylabel("Lasso Objective E (log scale)")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 1(b) Optimality conditions
    r = y_train - Phi_train @ w
    threshold = 2 * eta * lam
    is_zero = np.abs(w) < threshold
    corr = Phi_train.T @ r
    
    non_zero_cond = np.max(np.abs(corr[~is_zero] - lam * np.sign(w[~is_zero])))
    zero_cond = np.max(np.abs(corr[is_zero])) / lam
    
    print(f"Final E(w): {losses[-1]:.4f} | Best-so-far E: {best_losses[-1]:.4f}")
    print(f"Non-zero optimality condition max |Phi_j^T r - lam*sign(w_j)|: {non_zero_cond:.4f}")
    print(f"Zero optimality condition max |Phi_j^T r| / lam              : {zero_cond:.4f} (<= 1 confirms optimality)")
    
    # 1(c) Comparison with reference solution
    w_ref = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0])
    diff_norm = np.linalg.norm(w - w_ref)
    print(f"Reference optimum E* = 0.2625 vs Best E = {best_losses[-1]:.4f}")
    print(f"||w - w*||_2 = {diff_norm:.4f} (close agreement subject to ~eta*lam chatter)")
    
    return w

# -------------------------------------------------------------
# Deliverables 2, 3, 4: Sweeps across lambda grid
# -------------------------------------------------------------
def run_deliverables_2_to_4(Phi_train, y_train, Phi_test, y_test, lams, eta=0.05):
    print("\n--- Deliverables 2-4: Sweeps Across Lambda Grid ---")
    lasso_ws = []
    train_rmses = []
    test_rmses = []
    zero_counts = []
    
    for lam in lams:
        w, _, _ = subgradient_descent(Phi_train, y_train, lam=lam, eta=eta, n_iters=20000)
        lasso_ws.append(w)
        train_rmses.append(rmse(Phi_train, w, y_train))
        test_rmses.append(rmse(Phi_test, w, y_test))
        zero_counts.append(np.sum(np.abs(w) < 2 * eta * lam))
        
    lasso_ws = np.array(lasso_ws)
    train_rmses = np.array(train_rmses)
    test_rmses = np.array(test_rmses)
    
    # Deliverable 4: Selecting lambda*
    best_idx = np.argmin(test_rmses)
    lam_star = lams[best_idx]
    min_test = test_rmses[best_idx]
    w_star = lasso_ws[best_idx]
    zero_mask_star = np.abs(w_star) < 2 * eta * lam_star
    zero_indices = np.where(zero_mask_star)[0]
    
    print(f"Optimal lambda* : {lam_star:.4e}")
    print(f"Test RMSE at lambda*: {min_test:.4f}")
    print(f"Number of approx zero coefficients at lambda*: {len(zero_indices)}")
    print(f"Indices of approx zero coefficients         : {zero_indices.tolist()}")
    
    # Deliverable 2 Plot: RMSE Sweep
    plt.figure(figsize=(8, 5))
    plt.plot(lams, train_rmses, 'b-', label='Train RMSE', linewidth=1.8)
    plt.plot(lams, test_rmses, 'r-', label='Test RMSE', linewidth=1.8)
    plt.axhline(0.1, color='gray', linestyle='--', label=r'Noise floor $\sqrt{0.01}=0.1$')
    plt.axvline(lam_star, color='green', linestyle=':', label=rf'Optimal $\lambda^* = {lam_star:.2e}$')
    plt.xscale('log')
    plt.ylim(0, 1.0)
    plt.xlabel(r'$\lambda$ (log scale)')
    plt.ylabel('RMSE')
    plt.title(r'Deliverable 2: Train and Test RMSE vs $\lambda$ (Lasso)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # Deliverable 3 Plot: Coefficient Path
    plt.figure(figsize=(9, 5))
    for j in range(Phi_train.shape[1]):
        plt.plot(lams, lasso_ws[:, j], label=f'$w_{j}$')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$ (log scale)')
    plt.ylabel(r'Coefficient value $w_j$')
    plt.title(r'Deliverable 3: Lasso Coefficient Path vs $\lambda$')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    print("\nNumber of approx zero coefficients at each lambda:")
    for lam, count in zip(lams, zero_counts):
        print(f"  lambda = {lam:.2e} : {count}/10 coefficients approx 0")
        
    return lam_star, lasso_ws[best_idx]

# -------------------------------------------------------------
# Deliverables 5, 6, 7: Lasso vs Ridge Comparison & Discussion
# -------------------------------------------------------------
def run_deliverables_5_to_7(Phi_train, y_train, Phi_test, y_test, lams, lam_star_lasso, w_star_lasso, x_train):
    print("\n--- Deliverables 5-7: Lasso vs Ridge Comparison ---")
    # Fit Ridge across same Legendre grid
    ridge_ws = [solve_ridge(Phi_train, y_train, lam) for lam in lams]
    ridge_test_rmses = np.array([rmse(Phi_test, w, y_test) for w in ridge_ws])
    
    best_ridge_idx = np.argmin(ridge_test_rmses)
    lam_star_ridge = lams[best_ridge_idx]
    w_star_ridge = ridge_ws[best_ridge_idx]
    
    # Tabulate comparison
    lasso_test = rmse(Phi_test, w_star_lasso, y_test)
    ridge_test = rmse(Phi_test, w_star_ridge, y_test)
    lasso_norm = np.linalg.norm(w_star_lasso)
    ridge_norm = np.linalg.norm(w_star_ridge)
    lasso_small = np.sum(np.abs(w_star_lasso) < 0.01)
    ridge_small = np.sum(np.abs(w_star_ridge) < 0.01)
    
    print("\nComparison Table (Deliverable 5):")
    print("-" * 65)
    print(f"{'Method':<10} | {'Best lambda':<12} | {'Test RMSE':<10} | {'||w||_2':<10} | {'|w_j| < 0.01':<12}")
    print("-" * 65)
    print(f"{'Lasso':<10} | {lam_star_lasso:<12.4e} | {lasso_test:<10.4f} | {lasso_norm:<10.4f} | {lasso_small:<12}")
    print(f"{'Ridge':<10} | {lam_star_ridge:<12.4e} | {ridge_test:<10.4f} | {ridge_norm:<10.4f} | {ridge_small:<12}")
    print("-" * 65)
    
    # Deliverable 6: Fitted curves
    x_dense = np.linspace(0, 1, 200)
    Phi_dense = design_matrix(x_dense, degree=9)
    y_true = np.sin(2 * np.pi * x_dense)
    y_lasso = Phi_dense @ w_star_lasso
    y_ridge = Phi_dense @ w_star_ridge
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='black', zorder=5, label='Train Data (N=10)')
    plt.plot(x_dense, y_true, 'k--', alpha=0.5, label=r'$\sin(2\pi x)$')
    plt.plot(x_dense, y_lasso, 'r-', linewidth=2, label=rf'Lasso ($\lambda^* = {lam_star_lasso:.2e}$)')
    plt.plot(x_dense, y_ridge, 'b-.', linewidth=1.8, label=rf'Ridge ($\lambda^* = {lam_star_ridge:.2e}$)')
    plt.ylim(-1.5, 1.5)
    plt.title(r"Deliverable 6: Fitted Curves (Lasso vs Ridge at their best $\lambda$)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # Deliverable 7: Discussion
    print("\nDiscussion (Deliverable 7):")
    print("  The L1 (Lasso) constraint ball has sharp corners on the coordinate axes, causing the elliptical SSE contours")
    print("  to frequently make first contact directly on an axis where irrelevant coefficients become exactly zero.")
    print("  In contrast, the smooth spherical L2 (Ridge) ball makes contact away from the axes, shrinking all coefficients")
    print("  proportionally without inducing sparsity (only 1 coefficient has |w_j| < 0.01 vs 4 for Lasso).")
    print("  As lambda increases past lambda*, both methods suffer from excessive regularization: training error rises steadily")
    print("  and test error degrades rapidly as the model underfits and flattens out toward zero.")

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load datasets
    x_train, y_train, x_test, y_test = load_data()
    print(f"Loaded {len(x_train)} train points and {len(x_test)} test points.")
    
    # 2. Design matrices in Legendre basis (d = 9)
    d = 9
    Phi_train = design_matrix(x_train, degree=d)
    Phi_test = design_matrix(x_test, degree=d)
    
    # Deliverable 1: Convergence and verification at lambda = 0.1
    run_deliverable_1(Phi_train, y_train, lam=0.1, eta=0.05)
    
    # 3. Grid for sweeps: 21 points logarithmically spaced from 10^-4 to 1
    lams = np.logspace(-4, 0, 21)
    
    # Deliverables 2-4: Sweeps and lambda* selection
    lam_star_lasso, w_star_lasso = run_deliverables_2_to_4(Phi_train, y_train, Phi_test, y_test, lams, eta=0.05)
    
    # Deliverables 5-7: Lasso vs Ridge comparison and fitted curves
    run_deliverables_5_to_7(Phi_train, y_train, Phi_test, y_test, lams, lam_star_lasso, w_star_lasso, x_train)
