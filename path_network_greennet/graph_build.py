import math
import numpy as np
import networkx as nx
from shapely.geometry import LineString
import geopandas as gpd

def build_initial_graph(gdf):
    G = nx.Graph()
    for _, row in gdf.iterrows():
        node_id = row["cell_id"]
        G.add_node(
            node_id,
            cx=float(row["cx"]),
            cy=float(row["cy"]),
            total_score=float(row["total_score"]),
            cand_node=bool(row["cand_node"]),
            green_node=False,
            green_gateway=False,
        )
    return G


def add_candidate_edges(G, gdf, dist_thr_m=150.0):
    cand = gdf[gdf["cand_node"] == True].copy()
    cand_nodes = cand["cell_id"].tolist()
    cx = cand.set_index("cell_id")["cx"].to_numpy()
    cy = cand.set_index("cell_id")["cy"].to_numpy()
    eps = 1e-6

    try:
        from sklearn.neighbors import KDTree
        pts = np.vstack([cx, cy]).T
        kdt = KDTree(pts)
        pairs = set()
        for i, (xi, yi) in enumerate(pts):
            idxs = kdt.query_radius([[xi, yi]], r=dist_thr_m + eps)[0]
            for j in idxs:
                if j <= i:
                    continue
                pairs.add((i, j))
        for (i, j) in pairs:
            a = cand_nodes[i]
            b = cand_nodes[j]
            x1, y1 = float(cx[i]), float(cy[i])
            x2, y2 = float(cx[j]), float(cy[j])
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist <= dist_thr_m + eps:
                G.add_edge(a, b, weight=dist)

    except Exception:
        for i in range(len(cand_nodes)):
            xi, yi = float(cx[i]), float(cy[i])
            for j in range(i + 1, len(cand_nodes)):
                xj, yj = float(cy[j]), float(cy[j])
                xj, yj = float(cx[j]), float(cy[j])
                dist = math.hypot(xj - xi, yj - yi)
                if dist <= dist_thr_m + eps:
                    a = cand_nodes[i]
                    b = cand_nodes[j]
                    G.add_edge(a, b, weight=dist)

    return G


def minimum_spanning_subgraph(G):
    mst_edges = list(nx.minimum_spanning_edges(G, data=True))
    mstG = nx.Graph()
    mstG.add_nodes_from(G.nodes(data=True))
    for (u, v, d) in mst_edges:
        mstG.add_edge(u, v, weight=d["weight"])
    return mstG


def extract_rep_path(G, mstG):
    mst_edges = list(mstG.edges(data=True))
    paths = []
    for i in range(len(mst_edges)):
        for j in range(i + 1, len(mst_edges)):
            a, b, _ = mst_edges[i][0], mst_edges[i][1], mst_edges[i][2]
            c, d_, _ = mst_edges[j][0], mst_edges[j][1], mst_edges[j][2]
            for s, t in [(a, c), (a, d_), (b, c), (b, d_)]:
                try:
                    p = nx.shortest_path(mstG, s, t, weight="weight")
                    paths.append(p)
                except:
                    pass

    if not paths:
        comps = list(nx.connected_components(mstG))
        if not comps:
            return []
        comp_max = max(comps, key=len)
        comp_nodes = list(comp_max)
        if len(comp_nodes) >= 2:
            try:
                p = nx.shortest_path(mstG, comp_nodes[0], comp_nodes[-1], weight="weight")
                paths = [p]
            except:
                paths = [comp_nodes]
        else:
            paths = [comp_nodes]

    longest_path = max(paths, key=lambda pth: len(pth))
    return longest_path


def flag_green_nodes(G, path_nodes):
    for n in path_nodes:
        if n in G.nodes:
            G.nodes[n]["green_node"] = True
    return G


def ensure_path_edges(G, path_nodes):
    path_edges = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        path_edges.append((a, b))
        if not G.has_edge(a, b):
            x1, y1 = G.nodes[a]["cx"], G.nodes[a]["cy"]
            x2, y2 = G.nodes[b]["cx"], G.nodes[b]["cy"]
            dist = math.hypot(x2 - x1, y2 - y1)
            G.add_edge(a, b, weight=dist)
    return G, path_edges


def edges_to_gdf(G, path_edges, crs):
    line_geoms = []
    uu = []
    vv = []
    for (a, b) in path_edges:
        x1, y1 = G.nodes[a]["cx"], G.nodes[a]["cy"]
        x2, y2 = G.nodes[b]["cx"], G.nodes[b]["cy"]
        line = LineString([(x1, y1), (x2, y2)])
        line_geoms.append(line)
        uu.append(a)
        vv.append(b)

    edges_gdf = gpd.GeoDataFrame(
        {"u": uu, "v": vv, "geometry": line_geoms},
        crs=crs
    )
    return edges_gdf
