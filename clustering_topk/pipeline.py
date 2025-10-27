import os
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from math import ceil

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# optional dependencies
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except Exception:
    HAS_KMEDOIDS = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def split_columns(df: pd.DataFrame, id_cols: set) -> Tuple[List[str], List[str]]:
    num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_all = df.select_dtypes(exclude=np.number).columns.tolist()

    num_cols = [c for c in num_cols_all if c not in id_cols]
    cat_cols = [c for c in cat_cols_all if c not in id_cols]
    return num_cols, cat_cols


def standardize(df: pd.DataFrame, num_cols: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    assert len(num_cols) > 0, "수치형 피처가 없습니다. 전처리를 확인하세요."
    X = df[num_cols].astype(float).values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std, scaler


def fit_pca(
    X_std: np.ndarray,
    n_components: int,
    random_state: int,
) -> Tuple[PCA, np.ndarray, PCA, np.ndarray]:
    n_comp = min(n_components, X_std.shape[1])
    pca_model = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca_model.fit_transform(X_std)

    pca_full = PCA()
    X_full = pca_full.fit_transform(X_std)

    return pca_model, X_pca, pca_full, X_full


def get_algorithms(
    k_list: List[int],
    random_state: int,
) -> List[Tuple[str, object]]:
    algos: List[Tuple[str, object]] = []

    for k in k_list:
        algos.append(
            (f"KMeans(k={k})", KMeans(n_clusters=k, random_state=random_state, n_init="auto"))
        )
        algos.append(
            (
                f"MiniBatchKMeans(k={k})",
                MiniBatchKMeans(
                    n_clusters=k,
                    random_state=random_state,
                    batch_size=2048,
                    n_init="auto",
                ),
            )
        )
        algos.append(
            (
                f"GMM(k={k})",
                GaussianMixture(
                    n_components=k,
                    random_state=random_state,
                    covariance_type="full",
                ),
            )
        )
        if HAS_KMEDOIDS:
            algos.append(
                (
                    f"KMedoids(k={k})",
                    KMedoids(n_clusters=k, random_state=random_state, method="pam"),
                )
            )

    # density / hierarchical-ish
    algos.append(("DBSCAN", DBSCAN(eps=0.5, min_samples=10)))
    algos.append(("Birch", Birch(n_clusters=None)))
    if HAS_HDBSCAN:
        algos.append(("HDBSCAN", hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)))

    return algos


def _safe_scores(X_eval: np.ndarray, labels: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
        "n_clusters": np.nan,
        "noise_rate": np.nan,
    }

    if labels is None:
        return out

    noise_mask = labels != -1
    X_used = X_eval[noise_mask] if noise_mask is not None else X_eval
    y_used = labels[noise_mask] if noise_mask is not None else labels

    unique = np.unique(y_used)
    nclu = unique.size
    out["n_clusters"] = nclu
    if noise_mask is not None:
        out["noise_rate"] = 1 - np.sum(noise_mask) / len(labels)

    if nclu >= 2 and len(X_used) > nclu:
        try:
            out["silhouette"] = float(silhouette_score(X_used, y_used))
        except Exception:
            pass
        try:
            out["calinski_harabasz"] = float(calinski_harabasz_score(X_used, y_used))
        except Exception:
            pass
        try:
            out["davies_bouldin"] = float(davies_bouldin_score(X_used, y_used))
        except Exception:
            pass

    return out


def run_all_clusterings(
    X_for_fit: np.ndarray,
    algos: List[Tuple[str, object]],
    heavy_names: Tuple[str, ...],
    heavy_subsample_n: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rng = np.random.RandomState(random_state)
    n = X_for_fit.shape[0]

    rows = []
    labels_dict: Dict[str, np.ndarray] = {}

    for name, model in algos:
        X_train = X_for_fit
        fit_idx = None

        if any(h in name for h in heavy_names) and n > heavy_subsample_n:
            fit_idx = rng.choice(n, size=heavy_subsample_n, replace=False)
            X_train = X_for_fit[fit_idx]

        try:
            if hasattr(model, "fit_predict"):
                labels_partial = model.fit_predict(X_train)
            else:
                model.fit(X_train)
                if hasattr(model, "labels_"):
                    labels_partial = model.labels_
                elif hasattr(model, "predict"):
                    labels_partial = model.predict(X_train)
                else:
                    raise RuntimeError("라벨 추출 실패")

            if fit_idx is not None and hasattr(model, "predict"):
                try:
                    labels_full = model.predict(X_for_fit)
                except Exception:
                    labels_full = None
            elif fit_idx is not None:
                labels_full = None
            else:
                labels_full = labels_partial

            score_input_X = X_for_fit if labels_full is not None else X_train
            score_input_y = labels_full if labels_full is not None else labels_partial
            scores = _safe_scores(score_input_X, score_input_y)

            rows.append({"method": name, **scores})

            labels_dict[name] = score_input_y

            print(
                f"[OK] {name} -> clusters={scores['n_clusters']}, "
                f"silhouette={scores['silhouette']:.4f}"
            )

        except Exception as e:
            print(f"[SKIP] {name} 실패: {e}")
            rows.append(
                {
                    "method": name,
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "n_clusters": np.nan,
                    "noise_rate": np.nan,
                }
            )
            labels_dict[name] = None

    results_df = pd.DataFrame(rows).sort_values(by=["silhouette"], ascending=False)
    return results_df, labels_dict


def sweep_kmeans_over_k(
    X_for_fit: np.ndarray,
    k_list: List[int],
    random_state: int,
    silhouette_sample_n: int,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    rng = np.random.RandomState(random_state)

    n = X_for_fit.shape[0]
    sil_idx = np.arange(n)
    if n > silhouette_sample_n:
        sil_idx = rng.choice(n, size=silhouette_sample_n, replace=False)

    rows = []
    labels_by_k: Dict[int, np.ndarray] = {}

    for k in k_list:
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels_k = km.fit_predict(X_for_fit)
            inertia = float(km.inertia_)
            uniq = np.unique(labels_k)

            # metrics
            if uniq.size < 2:
                sil = np.nan
                ch = np.nan
                db = np.nan
            else:
                X_for_sil = X_for_fit[sil_idx]
                try:
                    sil = float(
                        silhouette_score(X_for_sil, labels_k[sil_idx])
                    )
                except Exception:
                    sil = np.nan
                try:
                    ch = float(calinski_harabasz_score(X_for_fit, labels_k))
                except Exception:
                    ch = np.nan
                try:
                    db = float(davies_bouldin_score(X_for_fit, labels_k))
                except Exception:
                    db = np.nan

            rows.append(
                {
                    "k": k,
                    "silhouette": sil,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "inertia": inertia,
                    "n_clusters": int(uniq.size),
                }
            )
            labels_by_k[k] = labels_k
            print(
                f"[OK] k={k:>2} sil={sil:.4f} CH={ch:.1f} DB={db:.4f} inertia={inertia:.1f}"
            )
        except Exception as e:
            print(f"[SKIP] k={k} 실패: {e}")
            rows.append(
                {
                    "k": k,
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "inertia": np.nan,
                    "n_clusters": np.nan,
                }
            )

    metrics_k = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return metrics_k, labels_by_k


def _best_by_sil(df: pd.DataFrame) -> int:
    if df["silhouette"].notna().any():
        return int(df.loc[df["silhouette"].idxmax(), "k"])
    return int(df.loc[df["inertia"].idxmin(), "k"])


def _best_by_db(df: pd.DataFrame) -> int:
    if df["davies_bouldin"].notna().any():
        return int(df.loc[df["davies_bouldin"].idxmin(), "k"])
    return _best_by_sil(df)


def _best_by_ch(df: pd.DataFrame) -> int:
    if df["calinski_harabasz"].notna().any():
        return int(df.loc[df["calinski_harabasz"].idxmax(), "k"])
    return _best_by_sil(df)


def choose_best_k(metrics_k: pd.DataFrame) -> Dict[str, int]:
    k_sil = _best_by_sil(metrics_k)
    k_db = _best_by_db(metrics_k)
    k_ch = _best_by_ch(metrics_k)

    best_summary = {
        "silhouette_max": k_sil,
        "davies_bouldin_min": k_db,
        "calinski_harabasz_max": k_ch,
    }
    return best_summary


def recover_cluster_centers_original_space(
    X_for_fit: np.ndarray,
    k_best: int,
    scaler: StandardScaler,
    pca_model: Optional[PCA],
    num_cols: List[str],
    random_state: int,
) -> Optional[pd.DataFrame]:
    try:
        km_best = KMeans(n_clusters=k_best, random_state=random_state, n_init="auto").fit(
            X_for_fit
        )
    except Exception:
        return None

    centers_work = km_best.cluster_centers_

    if pca_model is not None:
        centers_std = pca_model.inverse_transform(centers_work)
    else:
        centers_std = centers_work

    centers_orig = scaler.inverse_transform(centers_std)

    centers_df = pd.DataFrame(centers_orig, columns=num_cols)
    centers_df.insert(0, f"cluster_k{k_best}", range(k_best))
    return centers_df


def attach_best_labels(
    df: pd.DataFrame,
    labels_by_k: Dict[int, np.ndarray],
    k_best: int,
    prefix: str = "cluster_k",
) -> pd.DataFrame:
    df_out = df.copy()
    best_labels = labels_by_k[k_best]
    df_out[f"{prefix}{k_best}"] = best_labels
    return df_out


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)
