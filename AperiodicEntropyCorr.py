


import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.utils import resample

"""
Computes the Spearman and Pearson correlations between the aperiodic exponent
and Shannon entropy (H) of calcium fluorescence signals, across cell
populations and physiological conditions.

A non-parametric bootstrap procedure (B = 1,000 resamples with replacement)
is used to derive 95% confidence intervals for the Pearson correlation
coefficient.  Results are printed as a summary table and collected in a
DataFrame for downstream use.

Input file
----------
EntropyOfTheSignal.csv : semicolon-delimited table with at least the columns
    CellGroup, Condition, AperiodicValue, H.
"""


CONDITIONS  = ["Virgin", "Lactant", "Multipara", "Weaned", "OVX"]
POPULATIONS = ["Lactotrophs", "Somatotrophs", "All population"]
N_BOOTSTRAPS = 1_000
RANDOM_SEED  = 42


def load_data(path: str) -> pd.DataFrame:
    """Load and column-strip the entropy/aperiodic CSV file.

    Parameters
    ----------
    path : str
        Path to the semicolon-delimited input CSV.

    Returns
    -------
    pd.DataFrame
        Table with stripped column names.
    """
    df = pd.read_csv(path, delimiter=";")
    df.columns = df.columns.str.strip()
    return df


def bootstrap_pearson_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = 1_000,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Estimate a 95% bootstrap confidence interval for the Pearson correlation.

    Pairs (x, y) are resampled together with replacement to preserve their
    joint distribution.  The 2.5th and 97.5th percentiles of the bootstrap
    distribution of r are returned as the confidence interval.

    Parameters
    ----------
    x, y : np.ndarray
        Paired observations of equal length.
    n_bootstraps : int
        Number of bootstrap replicates (default 1,000).
    random_seed : int
        Random state for reproducibility (default 42).

    Returns
    -------
    tuple of (float, float)
        Lower and upper bounds of the 95% CI.
    """
    rng = np.random.default_rng(random_seed)
    boot_corrs = np.empty(n_bootstraps)

    for b in range(n_bootstraps):
        idx = rng.integers(0, len(x), size=len(x))
        r, _ = pearsonr(x[idx], y[idx])
        boot_corrs[b] = r

    return float(np.percentile(boot_corrs, 2.5)), float(np.percentile(boot_corrs, 97.5))


def correlate_group(
    df: pd.DataFrame,
    population: str,
    condition: str,
    n_bootstraps: int = 1_000,
    random_seed: int = 42,
) -> dict | None:
    """Compute Spearman and Pearson correlations for one population/condition.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with columns CellGroup, Condition, AperiodicValue, H.
    population : str
        Cell population label.
    condition : str
        Physiological condition label.
    n_bootstraps : int
        Bootstrap replicates for the Pearson CI (default 1,000).
    random_seed : int
        Random state for reproducibility (default 42).

    Returns
    -------
    dict or None
        Dictionary with keys: population, condition, spearman_r, spearman_p,
        pearson_r, pearson_p, ci_low, ci_high.
        Returns None if the subset contains fewer than 3 observations.
    """
    subset = df[
        (df["CellGroup"] == population) &
        (df["Condition"] == condition)
    ].dropna(subset=["AperiodicValue", "H"])

    if len(subset) < 3:
        return None

    aperiodic = subset["AperiodicValue"].values
    entropy   = subset["H"].values

    spearman_r, spearman_p = spearmanr(aperiodic, entropy)
    pearson_r,  pearson_p  = pearsonr(aperiodic,  entropy)
    ci_low, ci_high = bootstrap_pearson_ci(
        aperiodic, entropy, n_bootstraps, random_seed
    )

    return {
        "population": population,
        "condition":  condition,
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "pearson_r":  round(pearson_r,  4),
        "pearson_p":  round(pearson_p,  4),
        "ci_low":     round(ci_low,  4),
        "ci_high":    round(ci_high, 4),
    }


def run_correlation_analysis(
    df: pd.DataFrame,
    populations: list[str],
    conditions: list[str],
    n_bootstraps: int = 1_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run correlation analysis across all population/condition combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    populations : list of str
        Cell population labels to iterate over.
    conditions : list of str
        Physiological condition labels to iterate over.
    n_bootstraps : int
        Bootstrap replicates for Pearson CI (default 1,000).
    random_seed : int
        Random state (default 42).

    Returns
    -------
    pd.DataFrame
        One row per population/condition with correlation statistics.
    """
    records = []

    for population in populations:
        for condition in conditions:
            result = correlate_group(
                df, population, condition, n_bootstraps, random_seed
            )
            if result is None:
                print(f"Skipping {population} / {condition}: insufficient data.")
                continue
            records.append(result)
            print(
                f"{population:20s} | {condition:10s} | "
                f"Spearman r = {result['spearman_r']:+.3f}  p = {result['spearman_p']:.3f} | "
                f"Pearson  r = {result['pearson_r']:+.3f}  p = {result['pearson_p']:.3f} | "
                f"95% CI = [{result['ci_low']:.3f}, {result['ci_high']:.3f}]"
            )

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_data("EntropyOfTheSignal.csv")

    results = run_correlation_analysis(
        df,
        populations=POPULATIONS,
        conditions=CONDITIONS,
        n_bootstraps=N_BOOTSTRAPS,
        random_seed=RANDOM_SEED,
    )

    # Optional: save summary table
    results.to_csv("correlation_results.csv", index=False)
