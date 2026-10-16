import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    if mean is None:
        mean = np.array([2, -1])
    if covariance is None:
        covariance = np.array([[3, 1.2], [1.2, 1]])
    rng = np.random.default_rng(seed)
    # Generate Gaussian samples
    X = rng.multivariate_normal(mean, covariance, n_samples)
    return X

def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # Center
    mean = np.mean(X, axis=0)
    centered = X - mean

    # Covariance
    # np.cov() treates each row a variable, but our rows
    # are observations and column are features
    covariance = np.cov(centered, rowvar=False)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort by descending eigenvalues
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    components = eigenvectors[:, indices]

    # Project data
    scores = centered @ components

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    total = np.sum(eigenvalues)
    fractions = eigenvalues / total
    return fractions


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # two figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

    mean = result["mean"]
    components = result["components"]
    eigenvalues = result["eigenvalues"]

    # Data and principal components

    # make it translucent for observing dense area 
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.4)

    for i, color in enumerate(["red", "green"]):
      # standard deviation along the pca direction
      # * 2 to make it longer
      v = 2 * np.sqrt(eigenvalues[i]) * components[:, i]

      ax1.plot(
          [mean[0] - v[0], mean[0] + v[0]],
          [mean[1] - v[1], mean[1] + v[1]],
          color=color, linewidth=2, label=f"PC{i+1}"
      )

    ax1.set(title="Data with PCA Directions", xlabel="Feature 1", ylabel="Feature 2")
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(alpha=0.2)

    # variance
    fractions = explained_variance(eigenvalues)

    ax2.bar(["PC1", "PC2"], fractions * 100, color=["red", "green"])
    ax2.set(title="Explained Variance", ylabel="Explained Variance (%)", ylim=(0, 105))

    for i, f in enumerate(fractions):
        ax2.text(i, f * 100 + 2, f"{f:.1%}", ha="center")

    fig.tight_layout()
    return fig

if __name__ == '__main__':
    X = generate_data()
    result = pca(X)
    fig = make_plots(X, result)
    fig.savefig("pca.png", dpi=300, bbox_inches="tight")
