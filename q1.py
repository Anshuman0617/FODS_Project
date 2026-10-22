import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. Load Data & Helper Functions
# -------------------------------------------------------------
def load_data(filepath="Data_Sets/poly_train.csv"):
    df = pd.read_csv(filepath)
    return df['x'].values, df['y'].values

def design_matrix(x, degree):
    # Vandermonde matrix: [1, x, x^2, ..., x^d]
    return np.vander(x, N=degree + 1, increasing=True)

def sse_loss(Phi, w, t):
    residual = Phi @ w - t
    return 0.5 * np.sum(residual ** 2)

# -------------------------------------------------------------
# Deliverable 1: Closed-form fits for d = 1..9
# -------------------------------------------------------------
def run_deliverable_1(x_train, y_train):
    print("--- Running Deliverable 1 ---")
    x_dense = np.linspace(0, 1, 200)
    plt.figure(figsize=(12, 8))
    plt.scatter(x_train, y_train, color='black', zorder=5, label='Train Data')
    plt.plot(x_dense, np.sin(2 * np.pi * x_dense), 'k--', alpha=0.5, label='sin(2πx)')

    for d in range(1, 10):
        Phi = design_matrix(x_train, d)
        # Robust SVD solver as instructed
        w = np.linalg.lstsq(Phi, y_train, rcond=None)[0]
        
        Phi_dense = design_matrix(x_dense, d)
        y_pred = Phi_dense @ w
        plt.plot(x_dense, y_pred, label=f'd = {d}')

    plt.ylim(-1.5, 1.5)
    plt.title("Deliverable 1: Polynomial Fits (d = 1 to 9) vs Training Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# Deliverable 2 & 3: QR Gradient Descent Family (d = 9)
# -------------------------------------------------------------
def batch_gd_qr(Q, t, eta, epochs):
    d_plus_1 = Q.shape[1]
    alpha = np.zeros(d_plus_1)
    losses = []
    for _ in range(epochs):
        # Full sum over all N points, do NOT divide by N
        grad = Q.T @ (Q @ alpha - t)
        alpha -= eta * grad
        losses.append(0.5 * np.sum((Q @ alpha - t) ** 2))
    return alpha, np.array(losses)

def sgd_qr(Q, t, eta, epochs, seed=42):
    rng = np.random.default_rng(seed)
    N, d_plus_1 = Q.shape
    alpha = np.zeros(d_plus_1)
    losses = []
    for _ in range(epochs):
        indices = rng.permutation(N)
        for i in indices:
            q_i = Q[i]
            error = np.dot(q_i, alpha) - t[i]
            alpha -= eta * error * q_i
        losses.append(0.5 * np.sum((Q @ alpha - t) ** 2))
    return alpha, np.array(losses)

def minibatch_gd_qr(Q, t, eta, epochs, batch_size=3, seed=42):
    rng = np.random.default_rng(seed)
    N, d_plus_1 = Q.shape
    alpha = np.zeros(d_plus_1)
    losses = []
    for _ in range(epochs):
        indices = rng.permutation(N)
        # Batches of size 3, 3, 3, 1
        for start_idx in range(0, N, batch_size):
            b_idx = indices[start_idx:start_idx + batch_size]
            Q_b = Q[b_idx]
            t_b = t[b_idx]
            # Averaged gradient over mini-batch
            grad = (Q_b.T @ (Q_b @ alpha - t_b)) / len(b_idx)
            alpha -= eta * grad
        losses.append(0.5 * np.sum((Q @ alpha - t) ** 2))
    return alpha, np.array(losses)

def run_deliverables_2_and_3(Q, R, y_train, eta=0.1, epochs=2000):
    print("--- Running Deliverables 2 & 3: Loss Curves ---")
    alpha_batch, loss_batch = batch_gd_qr(Q, y_train, eta, epochs)
    alpha_sgd, loss_sgd = sgd_qr(Q, y_train, eta, epochs, seed=42)
    alpha_mb, loss_mb = minibatch_gd_qr(Q, y_train, eta, epochs, batch_size=3, seed=42)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_batch, label=f'Batch GD (η={eta})', linewidth=2)
    plt.plot(loss_sgd, '--', label=f'SGD (η={eta})', linewidth=2)
    plt.plot(loss_mb, label=f'Mini-batch GD (b=3, η={eta})', linewidth=2)
    plt.yscale('log')
    plt.title("Deliverable 3: Loss vs Epoch for GD Family (QR-reparameterised, d=9)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss E (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return alpha_batch, alpha_sgd, alpha_mb

# -------------------------------------------------------------
# Deliverable 4: Learning-Rate Study
# -------------------------------------------------------------
def run_deliverable_4(Q, R, y_train):
    print("--- Running Deliverable 4: Learning-Rate Study ---")
    # Regimes
    etas = {'Too small (η=0.01)': 0.01,
            'Well-chosen (η=0.1)': 0.1,
            'Too large (η=2.1)': 2.1}
    epochs = 100

    plt.figure(figsize=(10, 6))
    for label, eta in etas.items():
        alpha, loss = batch_gd_qr(Q, y_train, eta, epochs)
        plt.plot(range(1, epochs + 1), loss, label=f"{label}")
        if eta == 2.1:
            w_div = np.linalg.solve(R, alpha)
            print(f"\n[Divergent η=2.1 after 100 epochs]:")
            print(f"Final Loss E = {loss[-1]:.4e}")
            print("Weight vector w:")
            print(w_div)

    plt.yscale('log')
    plt.title("Deliverable 4: Batch GD Convergence for Different Learning Rates (100 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss E (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# Deliverable 5: Verification Against Closed-Form
# -------------------------------------------------------------
def run_deliverable_5(w_closed, R, alpha_batch, alpha_sgd, alpha_mb):
    print("\n--- Running Deliverable 5: Verification ---")
    w_batch = np.linalg.solve(R, alpha_batch)
    w_sgd = np.linalg.solve(R, alpha_sgd)
    w_mb = np.linalg.solve(R, alpha_mb)

    norm_closed = np.linalg.norm(w_closed, 2)
    variants = {'Batch GD': w_batch, 'SGD': w_sgd, 'Mini-batch GD': w_mb}

    for name, w_gd in variants.items():
        abs_err = np.linalg.norm(w_gd - w_closed, 2)
        rel_err = abs_err / norm_closed
        print(f"{name:15s} | Abs Error: {abs_err:.6e} | Rel Error: {rel_err:.6e}")

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data
    x_train, y_train = load_data("Data_Sets/poly_train.csv")
    print(f"Loaded {len(x_train)} training data points from Data_Sets/poly_train.csv.")

    # 2. Deliverable 1: Closed-form fits for d = 1..9
    run_deliverable_1(x_train, y_train)

    # 3. Setup for d = 9
    d = 9
    Phi_9 = design_matrix(x_train, d)
    w_closed = np.linalg.lstsq(Phi_9, y_train, rcond=None)[0]

    # Thin QR decomposition: Phi = Q @ R
    Q, R = np.linalg.qr(Phi_9)

    # 4. Deliverables 2 & 3: Run GD variants & plot loss curves
    alpha_batch, alpha_sgd, alpha_mb = run_deliverables_2_and_3(Q, R, y_train, eta=0.1, epochs=2000)

    # 5. Deliverable 4: Learning rate study
    run_deliverable_4(Q, R, y_train)

    # 6. Deliverable 5: Verification of GD vs Closed-form
    run_deliverable_5(w_closed, R, alpha_batch, alpha_sgd, alpha_mb)

