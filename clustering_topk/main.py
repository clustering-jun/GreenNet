import argparse
import os
import pandas as pd
import numpy as np

from config import ExperimentConfig
import pipeline
import viz


def run_experiment(cfg: ExperimentConfig):
    viz._set_pub_style(dpi=300, font_path=cfg.font_path)

    df = pipeline.load_dataframe(cfg.input_csv_path)

    num_cols, cat_cols = pipeline.split_columns(df, cfg.id_cols)

    X_std, scaler = pipeline.standardize(df, num_cols)

    pca_model, X_pca, pca_full, X_full = pipeline.fit_pca(
        X_std,
        n_components=cfg.pca_components,
        random_state=cfg.random_state,
    )

    if cfg.use_pca_for_plot2d:
        if X_pca.shape[1] >= 2:
            X_plot2d = X_pca[:, :2]
        else:
            from sklearn.decomposition import PCA
            X_plot2d = PCA(n_components=2, random_state=cfg.random_state).fit_transform(X_std)
    else:
        X_plot2d = X_std[:, :2]

    rng = np.random.RandomState(cfg.random_state)
    n_total = X_plot2d.shape[0]
    plot_idx = rng.choice(n_total, size=min(cfg.vis_sample_n, n_total), replace=False)

    if cfg.use_pca_for_clustering:
        X_for_cluster = X_pca
    else:
        X_for_cluster = X_std

    algos = pipeline.get_algorithms(
        k_list=cfg.k_list,
        random_state=cfg.random_state,
    )
    results_df, labels_dict = pipeline.run_all_clusterings(
        X_for_fit=X_for_cluster,
        algos=algos,
        heavy_names=cfg.heavy_names,
        heavy_subsample_n=cfg.heavy_subsample_n,
        random_state=cfg.random_state,
    )

    metrics_k, labels_by_k = pipeline.sweep_kmeans_over_k(
        X_for_fit=X_for_cluster,
        k_list=cfg.k_list,
        random_state=cfg.random_state,
        silhouette_sample_n=cfg.silhouette_sample_n,
    )

    best_dict = pipeline.choose_best_k(metrics_k)
    k_best = best_dict["silhouette_max"]

    df_with_best = pipeline.attach_best_labels(
        df=df,
        labels_by_k=labels_by_k,
        k_best=k_best,
        prefix="cluster_k",
    )

    centers_df_best = pipeline.recover_cluster_centers_original_space(
        X_for_fit=X_for_cluster,
        k_best=k_best,
        scaler=scaler,
        pca_model=pca_model if cfg.use_pca_for_clustering else None,
        num_cols=num_cols,
        random_state=cfg.random_state,
    )

    pipeline.ensure_dirs(cfg.output_dir_results, cfg.output_dir_figures)

    viz.plot_histograms_and_categories(
        df=df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        max_cats_to_plot=cfg.max_cats_to_plot,
        top_n=cfg.top_n,
        outdir=cfg.output_dir_figures,
        prefix="dist",
    )

    viz.plot_corr_heatmap(
        df=df,
        num_cols=num_cols,
        outdir=cfg.output_dir_figures,
        prefix="corr",
    )

    viz.plot_pca_scree(
        explained_var_ratio=pca_full.explained_variance_ratio_,
        outdir=cfg.output_dir_figures,
        prefix="pca_scree",
    )

    viz.plot_k_selection_curves(
        metrics_k=metrics_k,
        k_best=k_best,
        out_path=os.path.join(cfg.output_dir_figures, "k_selection_curves.png"),
    )

    topk_sorted = (
        metrics_k.dropna(subset=["silhouette"])
        .sort_values("silhouette", ascending=False)["k"]
        .tolist()
    )
    topk_sorted = topk_sorted[: cfg.plot_topn] if len(topk_sorted) >= cfg.plot_topn else topk_sorted

    for k in topk_sorted:
        labels_k = labels_by_k[k]
        out_sil_path = os.path.join(cfg.output_dir_figures, f"silhouette_k{k}.png")
        viz.plot_silhouette_per_cluster(
            X_for_fit=X_for_cluster,
            labels=labels_k,
            out_path=out_sil_path,
            title=f"Silhouette plot (k={k})",
        )

    out_cluster_grid = os.path.join(cfg.output_dir_figures, "clusters_grid.png")
    viz.plot_cluster_scatter_grid(
        X_2d=X_plot2d,
        labels_dict=labels_dict,
        sample_idx=plot_idx,
        out_path=out_cluster_grid,
    )

    if cfg.k_fixed_for_demo in labels_by_k:
        labels_demo = labels_by_k[cfg.k_fixed_for_demo]

        km_demo = None
        from sklearn.cluster import KMeans
        km_demo = KMeans(
            n_clusters=cfg.k_fixed_for_demo,
            random_state=cfg.random_state,
            n_init="auto",
        ).fit(X_for_cluster)
        centers_reduced = km_demo.cluster_centers_

        out_pairs_path = os.path.join(
            cfg.output_dir_figures, f"kmeans_pairs_k{cfg.k_fixed_for_demo}.png"
        )
        viz.plot_cluster_scatter_pairs(
            X_reduced=X_for_cluster,
            labels=labels_demo,
            centers_reduced=centers_reduced,
            out_path=out_pairs_path,
            max_components=cfg.pca_components,
        )

    results_df.to_csv(
        os.path.join(cfg.output_dir_results, "clustering_summary_all_algos.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    metrics_k.to_csv(
        os.path.join(cfg.output_dir_results, "kmeans_k_sweep_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    df_with_best.to_csv(
        os.path.join(
            cfg.output_dir_results, f"data_with_cluster_k{k_best}.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    if centers_df_best is not None:
        centers_df_best.to_csv(
            os.path.join(
                cfg.output_dir_results, f"cluster_centers_k{k_best}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

    id_like = None
    for cand in ["id", "ID", "Id", "gid", "GID", "uid", "UID", "code", "Code", "key", "Key"]:
        if cand in df.columns:
            id_like = cand
            break
    if id_like is None:
        id_like = next(
            (c for c in df.columns if c.lower().endswith("_id")),
            None,
        )

    if id_like is None:
        membership = pd.DataFrame(
            {
                "row_index": df.index.astype(int),
                "cluster": df_with_best[f"cluster_k{k_best}"].astype(int, errors="ignore"),
            }
        )
    else:
        membership = pd.DataFrame(
            {
                id_like: df[id_like].values,
                "cluster": df_with_best[f"cluster_k{k_best}"].astype(int, errors="ignore"),
            }
        )

    membership.to_csv(
        os.path.join(
            cfg.output_dir_results, f"cluster_membership_k{k_best}.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    print("=== DONE ===")
    print(f"- best k (silhouette): {k_best}")
    print(f"- figures saved to:   {cfg.output_dir_figures}")
    print(f"- csv saved to:       {cfg.output_dir_results}")


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Clustering experiment pipeline (paper-ready)")
    parser.add_argument("--data", type=str, default="./data.csv", help="input CSV path")
    parser.add_argument("--font", type=str, default=None, help="ttf font path for plots (optional)")
    parser.add_argument("--out_fig", type=str, default="./figures", help="figure output dir")
    parser.add_argument("--out_res", type=str, default="./results", help="results (csv) output dir")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        input_csv_path=args.data,
        font_path=args.font,
        output_dir_figures=args.out_fig,
        output_dir_results=args.out_res,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run_experiment(cfg)
