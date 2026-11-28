import numpy as np
import matplotlib.pyplot as plt


def repair_weights(w: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """
    Exact simplex projection with box constraints:
    Enforces w_i >= 0, w_i <= max_weight, and sum(w_i) == 1.0 simultaneously.
    """
    n = len(w)
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"max_weight ({max_weight}) * n_assets ({n}) must be >= 1.0.")

    w = np.maximum(w, 0.0)
    total = np.sum(w)
    w = (np.ones(n) / n) if total <= 1e-12 else (w / total)

    # Iterative water-filling to redistribute excess weight above max_weight
    while np.any(w > max_weight + 1e-12):
        excess = np.sum(np.maximum(w - max_weight, 0.0))
        w = np.minimum(w, max_weight)
        under_cap = w < max_weight - 1e-12
        if not np.any(under_cap):
            break
        w[under_cap] += excess * (w[under_cap] / np.sum(w[under_cap]))

    return w


def portfolio_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 2.0,
    risk_free_rate: float = 0.065,
    max_weight: float = 1.0,
) -> float:
    """
    Minimize Mean-Variance Utility: 0.5 * lambda * (w^T Cov w) - (w^T mu - rf).
    Lower values indicate superior risk-adjusted utility.
    """
    w = repair_weights(weights, max_weight)
    port_return = float(np.dot(w, expected_returns))
    port_variance = float(np.dot(w.T, np.dot(cov_matrix, w)))
    excess_return = port_return - risk_free_rate
    return 0.5 * risk_aversion * port_variance - excess_return


def grey_wolf_portfolio_optimizer(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_wolves: int = 60,
    n_iters: int = 250,
    risk_aversion: float = 2.0,
    risk_free_rate: float = 0.065,
    max_weight: float = 0.40,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n_assets = len(expected_returns)

    # Initialize pack uniformly across the constrained simplex
    raw_init = rng.dirichlet(np.ones(n_assets), size=n_wolves)
    wolves = np.array([repair_weights(w, max_weight) for w in raw_init])
    fitness = np.array([
        portfolio_objective(w, expected_returns, cov_matrix, risk_aversion, risk_free_rate, max_weight)
        for w in wolves
    ])

    alpha_pos, alpha_score = np.zeros(n_assets), np.inf
    beta_pos, beta_score = np.zeros(n_assets), np.inf
    delta_pos, delta_score = np.zeros(n_assets), np.inf

    def update_leaders(pop, fit):
        nonlocal alpha_pos, alpha_score, beta_pos, beta_score, delta_pos, delta_score
        for i in range(len(fit)):
            score = fit[i]
            pos = pop[i]
            if score < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = score, pos.copy()
            elif score < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = score, pos.copy()
            elif score < delta_score:
                delta_score, delta_pos = score, pos.copy()

    update_leaders(wolves, fitness)
    history = [alpha_score]

    for t in range(n_iters):
        # Nonlinear decay (exponent 1.5) devotes more steps to local exploitation
        a = 2.0 * (1.0 - t / max(n_iters - 1, 1)) ** 1.5

        for i in range(n_wolves):
            X = wolves[i]

            r1, r2 = rng.random(n_assets), rng.random(n_assets)
            X1 = alpha_pos - (2.0 * a * r1 - a) * np.abs(2.0 * r2 * alpha_pos - X)

            r1, r2 = rng.random(n_assets), rng.random(n_assets)
            X2 = beta_pos - (2.0 * a * r1 - a) * np.abs(2.0 * r2 * beta_pos - X)

            r1, r2 = rng.random(n_assets), rng.random(n_assets)
            X3 = delta_pos - (2.0 * a * r1 - a) * np.abs(2.0 * r2 * delta_pos - X)

            candidate = repair_weights((X1 + X2 + X3) / 3.0, max_weight)
            cand_score = portfolio_objective(
                candidate, expected_returns, cov_matrix, risk_aversion, risk_free_rate, max_weight
            )

            # Greedy selection preserves monotonic pack improvement
            if cand_score < fitness[i]:
                wolves[i] = candidate
                fitness[i] = cand_score

        update_leaders(wolves, fitness)
        history.append(alpha_score)

    best_w = alpha_pos
    best_ret = float(np.dot(best_w, expected_returns))
    best_vol = float(np.sqrt(np.dot(best_w.T, np.dot(cov_matrix, best_w))))
    sharpe = (best_ret - risk_free_rate) / best_vol if best_vol > 0 else 0.0

    # Risk attribution & diversification diagnostics
    marginal_risk = np.dot(cov_matrix, best_w) / best_vol
    risk_contrib = best_w * marginal_risk
    pct_risk_contrib = risk_contrib / best_vol
    asset_vols = np.sqrt(np.diag(cov_matrix))
    diversification_ratio = float(np.dot(best_w, asset_vols) / best_vol)

    return {
        "weights": best_w,
        "pct_risk_contribution": pct_risk_contrib,
        "expected_return": best_ret,
        "volatility": best_vol,
        "sharpe_ratio": sharpe,
        "diversification_ratio": diversification_ratio,
        "objective_value": alpha_score,
        "history": np.array(history),
    }


def plot_optimization_results(tickers, mu, cov, result, risk_free_rate=0.065, max_weight=0.40):
    """Generates a 3-panel diagnostic dashboard: Weights vs Risk, Frontier, and Convergence."""
    rng = np.random.default_rng(123)
    n_samples = 4000
    rand_w = np.array([
        repair_weights(w, max_weight)
        for w in rng.dirichlet(np.ones(len(tickers)), size=n_samples)
    ])
    rand_rets = (rand_w @ mu) * 100
    rand_vols = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov, rand_w)) * 100
    rand_sharpes = (rand_rets / 100 - risk_free_rate) / (rand_vols / 100)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle("Grey Wolf Portfolio Optimization — Diagnostic Dashboard", fontsize=14, fontweight="bold")

    # Panel 1: Capital Allocation vs. Risk Contribution
    x = np.arange(len(tickers))
    width = 0.36
    axes[0].bar(x - width / 2, result["weights"] * 100, width, label="Capital Weight (%)", color="#2563eb")
    axes[0].bar(x + width / 2, result["pct_risk_contribution"] * 100, width, label="Risk Contribution (%)", color="#f97316")
    axes[0].axhline(max_weight * 100, color="#dc2626", linestyle="--", linewidth=1, label=f"Max Cap ({max_weight*100:.0f}%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([t.replace(".NS", "") for t in tickers], rotation=20)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Allocation vs. Risk Attribution")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel 2: Feasible Risk-Return Space & GWO Optimum
    sc = axes[1].scatter(rand_vols, rand_rets, c=rand_sharpes, cmap="viridis", alpha=0.55, s=14)
    plt.colorbar(sc, ax=axes[1], label="Sharpe Ratio")
    opt_vol = result["volatility"] * 100
    opt_ret = result["expected_return"] * 100
    axes[1].scatter([opt_vol], [opt_ret], color="#dc2626", marker="*", s=260, edgecolor="black", zorder=5, label="GWO Optimal")
    axes[1].annotate(
        f"Optimal\n({opt_vol:.2f}%, {opt_ret:.2f}%)",
        xy=(opt_vol, opt_ret),
        xytext=(opt_vol - 1.1, opt_ret + 0.18),
        fontsize=9,
        fontweight="bold",
        color="#991b1b",
    )
    axes[1].set_xlabel("Annualized Volatility (%)")
    axes[1].set_ylabel("Expected Annual Return (%)")
    axes[1].set_title("Constrained Feasible Frontier")
    axes[1].legend(loc="lower right", frameon=False)
    axes[1].grid(alpha=0.25)

    # Panel 3: Convergence Curve
    axes[2].plot(result["history"], color="#059669", linewidth=2)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Objective Value (Lower = Better)")
    axes[2].set_title("GWO Convergence Trajectory")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    mu = np.array([0.16, 0.14, 0.13, 0.15, 0.17])
    vols = np.array([0.24, 0.22, 0.20, 0.25, 0.26])
    corr = np.array([
        [1.00, 0.35, 0.45, 0.30, 0.48],
        [0.35, 1.00, 0.32, 0.72, 0.30],
        [0.45, 0.32, 1.00, 0.28, 0.68],
        [0.30, 0.72, 0.28, 1.00, 0.27],
        [0.48, 0.30, 0.68, 0.27, 1.00],
    ])
    cov = np.outer(vols, vols) * corr

    MAX_CAP = 0.40  # Enforce 40% max single-stock exposure
    RF_RATE = 0.065

    result = grey_wolf_portfolio_optimizer(
        expected_returns=mu,
        cov_matrix=cov,
        n_wolves=60,
        n_iters=250,
        risk_aversion=2.0,
        risk_free_rate=RF_RATE,
        max_weight=MAX_CAP,
        seed=42,
    )

    print(f"{'Ticker':<14} | {'Weight':>8} | {'Risk Contrib':>12}")
    print("-" * 40)
    for ticker, w, rc in zip(tickers, result["weights"], result["pct_risk_contribution"]):
        print(f"{ticker:<14} | {w * 100:>7.2f}% | {rc * 100:>11.2f}%")

    print("-" * 40)
    print(f"Expected Annual Return : {result['expected_return'] * 100:.2f}%")
    print(f"Annualized Volatility  : {result['volatility'] * 100:.2f}%")
    print(f"Sharpe Ratio           : {result['sharpe_ratio']:.4f}")
    print(f"Diversification Ratio  : {result['diversification_ratio']:.4f}")
    print(f"Final Objective Value  : {result['objective_value']:.6f}")

    plot_optimization_results(tickers, mu, cov, result, risk_free_rate=RF_RATE, max_weight=MAX_CAP)
