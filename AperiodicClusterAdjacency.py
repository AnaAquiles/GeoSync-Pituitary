"""


Constructs and visualises a cell-by-cell aperiodic adjacency matrix for each
combination of cell population and physiological condition.

For each group, spectral cluster labels (from k-means classification) and
per-cell aperiodic exponents (from Lorentzian fitting) are merged, clusters
are balanced by downsampling, and a symmetric adjacency matrix is built where
diagonal entries encode each cell's own aperiodic exponent and off-diagonal
entries encode pairwise aperiodic similarity (arithmetic mean).  All entries
are z-score normalised before visualisation.

Input files
-----------
EntropyOFTheSignal.csv  : spectral cluster assignments, delimited by ';'.
                          Required columns: CellGroup, Condition, x, Cells, Cluster.
AperiodicValues.csv     : per-cell aperiodic exponents, delimited by ';'.
                          Required columns: Cell group, Condition, x, Cells, Aperiodic Value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.utils import resample

CONDITIONS  = ["Virgin", "OVX", "Lactant", "Weaned", "Multipara"]
POPULATIONS = ["Lactotrophs", "Somatotrophs", "All population"]
CLUSTERS    = [1, 2]
RANDOM_SEED = 42


def load_data(index_path: str, aperiodic_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and column-strip the two input CSV files.

    Parameters
    ----------
    index_path : str
        Path to the spectral cluster assignment CSV.
    aperiodic_path : str
        Path to the aperiodic exponent CSV.

    Returns
    -------
    index_df : pd.DataFrame
        Cluster assignment table.
    aperiodic_df : pd.DataFrame
        Aperiodic exponent table.
    """
    index_df    = pd.read_csv(index_path,    delimiter=";")
    aperiodic_df = pd.read_csv(aperiodic_path, delimiter=";")
    index_df.columns    = index_df.columns.str.strip()
    aperiodic_df.columns = aperiodic_df.columns.str.strip()
    return index_df, aperiodic_df


def merge_and_balance(
    index_df: pd.DataFrame,
    aperiodic_df: pd.DataFrame,
    population: str,
    condition: str,
    clusters: list[int],
    random_seed: int = 42,
) -> pd.DataFrame | None:
    """Filter, merge, and cluster-balance data for one population/condition.

    Cells absent from either dataset are dropped.  Clusters are downsampled
    without replacement to the size of the smallest cluster.

    Parameters
    ----------
    index_df : pd.DataFrame
        Full cluster assignment table.
    aperiodic_df : pd.DataFrame
        Full aperiodic exponent table.
    population : str
        Cell population label (e.g. 'Lactotrophs').
    condition : str
        Physiological condition label (e.g. 'Virgin').
    clusters : list of int
        Expected cluster identifiers.
    random_seed : int
        Random state for reproducible downsampling (default 42).

    Returns
    -------
    pd.DataFrame or None
        Balanced, sorted merged table, or None if insufficient data.
    """
    pop_index    = index_df[
        (index_df["CellGroup"] == population) &
        (index_df["Condition"] == condition)
    ]
    pop_aperiodic = aperiodic_df[
        (aperiodic_df["Cell group"] == population) &
        (aperiodic_df["Condition"] == condition)
    ]

    merged = pd.merge(
        pop_index[["x", "Cells", "Cluster"]],
        pop_aperiodic[["x", "Cells", "Aperiodic Value"]],
        on=["x", "Cells"],
        how="inner",
    )

    if merged.empty:
        return None

    min_size = merged.groupby("Cluster").size().min()

    balanced_parts = []
    for cluster in clusters:
        subset = merged[merged["Cluster"] == cluster]
        if subset.empty:
            continue
        if len(subset) > min_size:
            subset = resample(subset, replace=False, n_samples=min_size,
                              random_state=random_seed)
        balanced_parts.append(subset)

    if not balanced_parts:
        return None

    balanced = pd.concat(balanced_parts).sort_values(["Cluster", "Cells"])
    return balanced


def build_adjacency_matrix(balanced: pd.DataFrame) -> np.ndarray:
    """Build a symmetric aperiodic adjacency matrix from balanced cell data.

    Diagonal entries are each cell's own aperiodic exponent.  Off-diagonal
    entries are the arithmetic mean of the two cells' exponents, providing a
    symmetric pairwise similarity measure.  All entries are then z-score
    normalised across the full matrix.

    Parameters
    ----------
    balanced : pd.DataFrame
        Balanced merged table with columns 'Cells' and 'Aperiodic Value'.

    Returns
    -------
    np.ndarray, shape (n_cells, n_cells)
        Z-score normalised adjacency matrix.
    """
    cells     = balanced["Cells"].unique()
    n_cells   = len(cells)
    cell_map  = {cell: idx for idx, cell in enumerate(cells)}

    # Lookup arrays for fast indexing
    aperiodic = balanced.set_index("Cells")["Aperiodic Value"].to_dict()

    adj = np.full((n_cells, n_cells), np.nan)

    for cell_i, i in cell_map.items():
        for cell_j, j in cell_map.items():
            val_i = aperiodic.get(cell_i)
            val_j = aperiodic.get(cell_j)
            if val_i is None or val_j is None:
                continue
            adj[i, j] = val_i if i == j else (val_i + val_j) / 2.0

    valid = ~np.isnan(adj)
    if np.any(valid):
        adj_norm = np.where(valid, zscore(adj, nan_policy="omit"), np.nan)
    else:
        adj_norm = adj

    return adj_norm


def plot_adjacency_matrix(
    adj_norm: np.ndarray,
    balanced: pd.DataFrame,
    population: str,
    condition: str,
    clusters: list[int],
) -> None:
    """Plot the lower-triangular aperiodic adjacency matrix as a heatmap.

    Cluster boundaries are annotated along both axes.

    Parameters
    ----------
    adj_norm : np.ndarray, shape (n_cells, n_cells)
        Z-score normalised adjacency matrix.
    balanced : pd.DataFrame
        Balanced merged table used to build the matrix (for cluster labels).
    population : str
        Cell population label (used in the figure title).
    condition : str
        Physiological condition label (used in the figure title).
    clusters : list of int
        Cluster identifiers (used for the legend).
    """
    n_cells = adj_norm.shape[0]
    upper_mask = np.triu(np.ones_like(adj_norm, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        adj_norm,
        mask=upper_mask,
        cmap=cmap,
        center=0,
        vmin=-3,
        vmax=3,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Normalised aperiodic value (z)"},
        ax=ax,
    )

    # Cluster boundary annotations
    cluster_labels = balanced.set_index("Cells")["Cluster"].values
    boundaries = np.where(np.diff(cluster_labels) != 0)[0] + 1
    boundaries = np.insert(boundaries, 0, 0)

    tick_positions = [
        (boundaries[i] + boundaries[i + 1]) / 2
        if i < len(boundaries) - 1
        else (boundaries[i] + n_cells) / 2
        for i in range(len(boundaries))
    ]
    tick_labels = [
        f"Cluster {cluster_labels[boundaries[i]]}"
        for i in range(len(boundaries))
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=12)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=12)

    cluster_colors = {1: "red", 2: "blue"}
    for cluster in clusters:
        ax.scatter([], [], c=cluster_colors.get(cluster, "grey"),
                   label=f"Cluster {cluster}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))

    ax.set_title(f"Aperiodic adjacency matrix — {population}, {condition}",
                 fontsize=16)
    ax.set_xlabel("Cells", fontsize=14)
    ax.set_ylabel("Cells", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


if __name__ == "__main__":
    index_df, aperiodic_df = load_data(
        "EntropyOFTheSignal.csv",
        "AperiodicValues.csv",
    )

    for population in POPULATIONS:
        for condition in CONDITIONS:
            balanced = merge_and_balance(
                index_df, aperiodic_df,
                population, condition,
                CLUSTERS, RANDOM_SEED,
            )

            if balanced is None:
                print(f"Skipping {population} / {condition}: insufficient data.")
                continue

            adj_norm = build_adjacency_matrix(balanced)
            plot_adjacency_matrix(adj_norm, balanced, population, condition, CLUSTERS)