from dataclasses import dataclass, field
from typing import List, Set, Tuple

@dataclass
class ExperimentConfig:
    input_csv_path: str = "./data.csv"
    id_cols: Set[str] = field(default_factory=lambda: {"geometry", "id_total_pop"})

    random_state: int = 42
    pca_components: int = 4
    use_pca_for_clustering: bool = True
    use_pca_for_plot2d: bool = True
    vis_sample_n: int = 8000

    k_list: List[int] = field(default_factory=lambda: list(range(2, 17)))
    k_fixed_for_demo: int = 4

    heavy_names: Tuple[str, ...] = ("GMM",)
    heavy_subsample_n: int = 20000

    silhouette_sample_n: int = 20000

    top_n: int = 10
    max_cats_to_plot: int = 30
    plot_topn: int = 3

    output_dir_results: str = "./results"
    output_dir_figures: str = "./figures"

    font_path: str = None