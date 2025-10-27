import math
import matplotlib.pyplot as plt
from io_utils import load_cells, export_results
from graph_build import (
    build_initial_graph,
    add_candidate_edges,
    minimum_spanning_subgraph,
    extract_rep_path,
    flag_green_nodes,
    ensure_path_edges,
    edges_to_gdf,
)
from metrics import (
    compute_node_metrics,
    compute_edge_metrics,
    fragmentation_test,
)

IN_GPKG  = "top200_all_cell.gpkg"
OUT_GPKG = "top200_all_cell_green_net.gpkg"
LAYER    = None
DIST_THR = 150.0
EDGE_REMOVE_RATIO = 0.2

print("Input:", IN_GPKG)
print("Output:", OUT_GPKG)

gdf = load_cells(IN_GPKG, layer=LAYER)
print("셀 개수:", len(gdf))
print("CRS:", gdf.crs)

G = build_initial_graph(gdf)
G = add_candidate_edges(G, gdf, dist_thr_m=DIST_THR)
print("초기 그래프 노드/간선:", len(G.nodes), len(G.edges))

mstG = minimum_spanning_subgraph(G)
print("MST 노드/간선:", len(mstG.nodes), len(mstG.edges))

rep_path = extract_rep_path(G, mstG)
print("대표 경로 길이(노드 수):", len(rep_path))

G = flag_green_nodes(G, set(rep_path))
G, path_edges = ensure_path_edges(G, rep_path)

gdf, = (gdf,)
gdf["green_node"] = gdf["cell_id"].map(lambda nid: G.nodes[nid]["green_node"] if nid in G.nodes else False)

edges_gdf = edges_to_gdf(G, path_edges, crs=gdf.crs)

G, gdf, subG = compute_node_metrics(G, gdf)
edges_gdf = compute_edge_metrics(subG, edges_gdf)

frag_stats = fragmentation_test(subG, edges_gdf, ratio_top=EDGE_REMOVE_RATIO)

print("게이트웨이 노드 상위:")
print(
    gdf[gdf["green_node"]]
    .sort_values("deg_cen", ascending=False)
    .head(10)[["cell_id","btw_cen","deg_cen","green_gateway"]]
)

print("중심성 상위:")
print(
    gdf[gdf["green_node"]]
    .sort_values("btw_cen", ascending=False)
    .head(10)[["cell_id","btw_cen","deg_cen","green_gateway"]]
)

print("엣지 중심성 상위:")
print(
    edges_gdf.sort_values("edge_btw", ascending=False)
    .head(10)[["u","v","edge_btw","is_bridge","critical"]]
)

print("fragmentation:")
print(f"  ratio_top              : {frag_stats['k_ratio']*100:.1f}%")
print(f"  num_comp_before_remove : {frag_stats['num_comp_before']}")
print(f"  num_comp_after_remove  : {frag_stats['num_comp_after']}")

plt.figure(figsize=(6,6))
ax = plt.gca()
gdf.plot(ax=ax, color="lightgray", alpha=0.4)
gdf[gdf["green_node"]].plot(ax=ax, color="green", markersize=20)
edges_gdf.plot(ax=ax, color="blue", linewidth=2)
plt.title("녹지 네트워크 (centroid 기반)")
plt.show()

out_written = export_results(gdf, edges_gdf, OUT_GPKG)
print("저장 완료:", out_written)
