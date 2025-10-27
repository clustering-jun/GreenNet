import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import silhouette_samples


def _set_pub_style(dpi=300, font_path: Optional[str] = None):
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
        }
    )
    if font_path is not None:
        from matplotlib import font_manager
        fontprop = font_manager.FontProperties(fname=font_path)
        mpl.rcParams["font.family"] = fontprop.get_name()
        mpl.rcParams["axes.unicode_minus"] = False


def plot_histograms_and_categories(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    max_cats_to_plot: int,
    top_n: int,
    outdir: str,
    prefix: str = "dist",
):
    n_numeric = len(num_cols)
    if n_numeric > 0:
        ncols = 3
        nrows = int(np.ceil(n_numeric / ncols))
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.8 * nrows)
        )
        if n_numeric == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        use_log_for = set()
        last_i = -1
        bins = 30

        for i, col in enumerate(num_cols):
            ax = axes_flat[i]
            s = df[col].dropna()
            auto_log = False
            if (s > 0).all() and s.max() > s.quantile(0.75) * 50:
                auto_log = True
            do_log = (col in use_log_for) or auto_log
            to_plot = np.log1p(s) if do_log else s
            label_hist = f"{col}" + (" (log1p)" if do_log else "")

            ax.hist(to_plot, bins=bins, density=True, alpha=0.55, edgecolor="black")
            try:
                to_plot.plot(kind="kde", ax=ax, lw=2)
            except Exception:
                pass

            ax.set_title(f"[Hist+KDE] {label_hist}")
            ax.set_xlabel(label_hist)
            ax.set_ylabel("Density")
            last_i = i

        for j in range(last_i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"{prefix}_numeric.png"), bbox_inches="tight")
        plt.close(fig)

    if len(cat_cols) > 0:
        plot_cols = []
        skipped = []
        for c in cat_cols:
            if df[c].nunique(dropna=False) > max_cats_to_plot:
                skipped.append(c)
            else:
                plot_cols.append(c)

        if len(plot_cols) > 0:
            n_cats = len(plot_cols)
            ncols = 2
            nrows = int(np.ceil(n_cats / ncols))
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3.8 * nrows)
            )
            if n_cats == 1:
                axes = np.array([[axes]])
            elif nrows == 1:
                axes = np.array([axes])
            axes_flat = axes.flatten()

            last_i = -1
            for i, col in enumerate(plot_cols):
                ax = axes_flat[i]
                vc = (
                    df[col]
                    .astype("string")
                    .fillna("<NA>")
                    .value_counts()
                    .head(top_n)
                )
                vc.plot(kind="barh", ax=ax)
                ax.invert_yaxis()
                ax.set_title(f"[Bar] {col} (Top {top_n})")
                ax.set_xlabel("Count")
                ax.set_ylabel(col)
                last_i = i
            for j in range(last_i + 1, len(axes_flat)):
                axes_flat[j].axis("off")

            plt.tight_layout()
            fig.savefig(os.path.join(outdir, f"{prefix}_categorical.png"), bbox_inches="tight")
            plt.close(fig)


def plot_corr_heatmap(
    df: pd.DataFrame,
    num_cols: List[str],
    outdir: str,
    prefix: str = "corr",
):
    if len(num_cols) < 2:
        return

    corr = df[num_cols].corr(method="pearson").values
    fig, ax = plt.subplots(
        figsize=(0.7 * len(num_cols) + 4.5, 0.7 * len(num_cols) + 4.5), dpi=300
    )
    ax.grid(False)
    im = ax.imshow(
        corr, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest"
    )
    ax.set_title("[Correlation Heatmap] (Pearson)")
    ax.set_xticks(np.arange(len(num_cols)))
    ax.set_yticks(np.arange(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)

    # value annotate (작을 때만)
    if len(num_cols) <= 15:
        for (r, c), val in np.ndenumerate(corr):
            ax.text(
                c,
                r,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{prefix}_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def plot_pca_scree(
    explained_var_ratio: np.ndarray,
    outdir: str,
    prefix: str = "pca_scree",
):
    """PCA Scree Plot + 누적분산"""
    evr = explained_var_ratio
    cum = np.cumsum(evr)
    idx = np.arange(1, len(evr) + 1)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(idx, evr, marker="o", linewidth=2, label="Explained variance ratio")
    plt.plot(
        idx,
        cum,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Cumulative explained variance",
    )
    plt.title("Scree Plot of PCA", pad=15)
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Ratio")
    plt.xticks(idx)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="best")
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{prefix}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_per_cluster(
    X_for_fit: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
):
    try:
        sil_values = silhouette_samples(X_for_fit, labels)
    except Exception:
        return

    order = np.argsort(labels, kind="stable")
    sv = sil_values[order]
    labs = labels[order]

    y_lower = 10
    fig = plt.figure(figsize=(6.5, 4.0))
    for c in np.unique(labs):
        sv_c = sv[labs == c]
        size_c = len(sv_c)
        y_upper = y_lower + size_c
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            sv_c,
            alpha=0.7,
        )
        plt.text(-0.05, y_lower + 0.5 * size_c, str(c))
        y_lower = y_upper + 10

    plt.axvline(np.mean(sil_values), linestyle="--")
    plt.title(title)
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter_grid(
    X_2d: np.ndarray,
    labels_dict: Dict[str, Optional[np.ndarray]],
    sample_idx: np.ndarray,
    out_path: str,
):
    methods = list(labels_dict.keys())
    nplots = len(methods)
    if nplots == 0:
        return

    ncols = 3
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows)
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes_flat = axes.flatten()
    last_i = -1

    for i, name in enumerate(methods):
        ax = axes_flat[i]
        y = labels_dict[name]
        if y is None:
            ax.scatter(
                X_2d[sample_idx, 0],
                X_2d[sample_idx, 1],
                s=3,
                alpha=0.7,
            )
            ax.set_title(f"{name}\n(no labels)", wrap=True)
        else:
            ax.scatter(
                X_2d[sample_idx, 0],
                X_2d[sample_idx, 1],
                c=y[sample_idx],
                s=3,
                alpha=0.7,
            )
            ax.set_title(name, wrap=True)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        last_i = i

    for j in range(last_i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter_pairs(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    centers_reduced: Optional[np.ndarray],
    out_path: str,
    max_components: int = 4,
):
    n_comp = min(max_components, X_reduced.shape[1])
    comp_names = [f"PC{i+1}" for i in range(n_comp)]
    pairs = list(combinations(range(n_comp), 2))

    nplots = len(pairs)
    ncols = 3
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows)
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    axes_flat = axes.flatten()
    last_i = -1

    for i, (a, b) in enumerate(pairs):
        ax = axes_flat[i]
        ax.scatter(
            X_reduced[:, a],
            X_reduced[:, b],
            c=labels,
            s=6,
            alpha=0.7,
        )
        if centers_reduced is not None and centers_reduced.shape[1] > max(a, b):
            ax.scatter(
                centers_reduced[:, a],
                centers_reduced[:, b],
                s=160,
                marker="X",
                edgecolor="black",
            )
        ax.set_xlabel(comp_names[a])
        ax.set_ylabel(comp_names[b])
        ax.set_title(f"{comp_names[a]} vs {comp_names[b]}")
        last_i = i

    for j in range(last_i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_k_selection_curves(
    metrics_k: pd.DataFrame,
    k_best: int,
    out_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes.ravel()

    # Inertia
    ax[0].plot(metrics_k["k"], metrics_k["inertia"], marker="o")
    ax[0].set_title("Elbow (Inertia)")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Inertia")
    ax[0].axvline(k_best, linestyle="--")

    # Silhouette
    ax[1].plot(metrics_k["k"], metrics_k["silhouette"], marker="o")
    ax[1].set_title("Silhouette (higher is better)")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Silhouette")
    ax[1].axvline(k_best, linestyle="--")

    # Calinski-Harabasz
    ax[2].plot(metrics_k["k"], metrics_k["calinski_harabasz"], marker="o")
    ax[2].set_title("Calinski-Harabasz (higher is better)")
    ax[2].set_xlabel("k")
    ax[2].set_ylabel("CH Index")
    ax[2].axvline(k_best, linestyle="--")

    # Davies-Bouldin
    ax[3].plot(metrics_k["k"], metrics_k["davies_bouldin"], marker="o")
    ax[3].set_title("Davies-Bouldin (lower is better)")
    ax[3].set_xlabel("k")
    ax[3].set_ylabel("DB Index")
    ax[3].axvline(k_best, linestyle="--")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
