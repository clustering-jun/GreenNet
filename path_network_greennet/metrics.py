import math
import networkx as nx
import pandas as pd

def compute_node_metrics(G, gdf):
    green_nodes = [nid for nid in G.nodes if G.nodes[nid].get("green_node", False)]
    subG = G.subgraph(green_nodes).copy()

    btw = nx.betweenness_centrality(subG, weight="weight", normalized=True)
    deg = dict(subG.degree())

    for n, v in btw.items():
        G.nodes[n]["btw_cen"] = v
    for n, v in deg.items():
        G.nodes[n]["deg_cen"] = v

    gdf["green_node"] = gdf["cell_id"].map(lambda nid: G.nodes[nid]["green_node"] if nid in G.nodes else False)
    gdf["btw_cen"] = gdf["cell_id"].map(lambda nid: G.nodes[nid].get("btw_cen", 0.0) if nid in G.nodes else 0.0)
    gdf["deg_cen"] = gdf["cell_id"].map(lambda nid: G.nodes[nid].get("deg_cen", 0) if nid in G.nodes else 0).astype(int)

    if len(green_nodes) > 0:
        deg_ranked = sorted(green_nodes, key=lambda nid: G.nodes[nid].get("deg_cen", 0), reverse=True)
        gateway = deg_ranked[0]
        G.nodes[gateway]["green_gateway"] = True
        gdf["green_gateway"] = gdf["cell_id"].map(lambda nid: G.nodes[nid].get("green_gateway", False) if nid in G.nodes else False)
    else:
        gdf["green_gateway"] = False

    return G, gdf, subG


def compute_edge_metrics(subG, edges_gdf):
    edge_btw = nx.edge_betweenness_centrality(subG, normalized=True, weight="weight")

    vals = []
    bridges = set(nx.bridges(subG))
    crit_list = []
    for (u, v) in zip(edges_gdf["u"], edges_gdf["v"]):
        key = tuple(sorted([u, v]))
        vals.append(edge_btw.get(key, 0.0))

        if subG.has_edge(u, v):
            c0 = nx.number_connected_components(subG)
            tmp = subG.copy()
            tmp.remove_edge(u, v)
            c1 = nx.number_connected_components(tmp)
            crit_list.append(c1 > c0)
        else:
            crit_list.append(False)

    edges_gdf["edge_btw"] = vals
    edges_gdf["is_bridge"] = [
        (tuple(sorted([u, v])) in bridges)
        for (u, v) in zip(edges_gdf["u"], edges_gdf["v"])
    ]
    edges_gdf["critical"] = crit_list

    return edges_gdf


def fragmentation_test(subG, edges_gdf, ratio_top=0.2):
    if len(edges_gdf) == 0:
        return {
            "num_comp_before": nx.number_connected_components(subG),
            "num_comp_after": nx.number_connected_components(subG),
            "k_ratio": ratio_top,
        }

    thr = edges_gdf["edge_btw"].quantile(1 - ratio_top)

    before = nx.number_connected_components(subG)
    subG2 = subG.copy()

    removable = edges_gdf.loc[edges_gdf["edge_btw"] >= thr, ["u", "v"]]
    for (a, b) in removable.itertuples(index=False, name=None):
        if subG2.has_edge(a, b):
            subG2.remove_edge(a, b)

    after = nx.number_connected_components(subG2)

    return {
        "num_comp_before": before,
        "num_comp_after": after,
        "k_ratio": ratio_top,
    }
