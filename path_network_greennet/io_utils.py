import os
import math
import pandas as pd
import geopandas as gpd

def load_cells(in_path, layer=None):
    gdf = gpd.read_file(in_path, layer=layer) if layer else gpd.read_file(in_path)

    if "topk_point" not in gdf.columns:
        raise ValueError("컬럼 'topk_point'가 없습니다.")
    if "total_score" not in gdf.columns:
        raise ValueError("컬럼 'total_score'가 없습니다.")

    gdf["topk_point"] = pd.to_numeric(gdf["topk_point"], errors="coerce").fillna(0).astype(int)
    gdf["total_score"] = pd.to_numeric(gdf["total_score"], errors="coerce").fillna(0.0).astype(float)

    if gdf.crs is None:
        raise ValueError("입력 데이터에 CRS(좌표계)가 없습니다. 미터 단위 projected CRS여야 합니다.")

    gdf["centroid"] = gdf.geometry.centroid
    gdf["cx"] = gdf["centroid"].x
    gdf["cy"] = gdf["centroid"].y

    if "cell_id" not in gdf.columns:
        gdf["cell_id"] = range(len(gdf))
    gdf["cell_id"] = gdf["cell_id"].astype(str)

    gdf["cand_node"] = gdf["topk_point"] == 1
    gdf["green_node"] = False
    gdf["green_gateway"] = False

    return gdf


def export_results(gdf_cells, gdf_edges, out_path, layer_cells="cells", layer_edges="green_edges"):
    gdf_out = gdf_cells.drop(columns=["centroid"])
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass
    gdf_out.to_file(out_path, driver="GPKG", layer=layer_cells)
    gdf_edges.to_file(out_path, driver="GPKG", layer=layer_edges)
    return out_path
