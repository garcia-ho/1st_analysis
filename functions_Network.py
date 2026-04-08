import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import networkx as nx



# =============================================================================
# 1. Correlation Matrix
# =============================================================================

label_map = {
    "ses_factor1_score": "SES1",
    "ses_factor2_score": "SES2",
    "ses_index": "Overall SES",
    "conceptual_exposure_score": "Conceptual exposure",
    "practical_ai_use_score": "Practical AI use",
    "learning_ecology_score": "Learning ecology",
    "language_load_score": "Language load",
    "epistemic_stance_score": "Epistemic stance",
    "ai_factor1_score": "AI understanding",
    "ai_factor2_score": "AI confidence",
        "ai_lit_score": "AI literacy",
}


def partial_corr_one(df, x, y, controls):
    sub = df[[x, y] + controls].dropna().copy()

    if len(controls) == 0:
        r, p = stats.pearsonr(sub[x], sub[y])
        return r, p, len(sub)

    Xc = sm.add_constant(sub[controls])

    model_x = sm.OLS(sub[x], Xc).fit()
    model_y = sm.OLS(sub[y], Xc).fit()

    resid_x = model_x.resid
    resid_y = model_y.resid

    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(sub)

def partial_corr_matrix(df, vars):
    dat = df[vars].dropna().copy()

    p = len(vars)
    est = pd.DataFrame(np.eye(p), index=vars, columns=vars, dtype=float)
    pval = pd.DataFrame(np.zeros((p, p)), index=vars, columns=vars, dtype=float)
    nmat = pd.DataFrame(np.full((p, p), len(dat)), index=vars, columns=vars, dtype=float)

    for i, x in enumerate(vars):
        for j, y in enumerate(vars):
            if i >= j:
                continue

            controls = [v for v in vars if v not in [x, y]]
            r, p, n = partial_corr_one(dat, x, y, controls)

            est.loc[x, y] = r
            est.loc[y, x] = r
            pval.loc[x, y] = p
            pval.loc[y, x] = p
            nmat.loc[x, y] = n
            nmat.loc[y, x] = n

    return {
        "estimate": est,
        "p_value": pval,
        "n": nmat,
        "data": dat
    }

def partial_corr_table(pc_obj, upper_only=True):

    est = pc_obj["estimate"]
    pval = pc_obj["p_value"]
    nmat = pc_obj["n"]

    rows = []
    vars = list(est.index)

    for i, v1 in enumerate(vars):
        for j, v2 in enumerate(vars):
            if upper_only and i >= j:
                continue
            if not upper_only and i == j:
                continue

            rows.append({
                "var1": v1,
                "var2": v2,
                "partial_r": est.loc[v1, v2],
                "p_value": pval.loc[v1, v2],
                "n": nmat.loc[v1, v2],
            })

    return pd.DataFrame(rows)

# multiple testing correction
def adjust_partial_corr_p(pc_table, method="fdr_bh", alpha=0.05):
    out = pc_table.copy()

    reject, p_adj, _, _ = multipletests(out["p_value"], alpha=alpha, method=method)

    out["p_adj"] = p_adj
    out["sig"] = reject
    return out


# =============================================================================
# 2. Network Analysis
# =============================================================================
# mapping for network
node_groups = {
    "SES1": "SES",
    "SES2": "SES",
    "Conceptual exposure": "Mediator",
    "Practical AI use": "Mediator",
    "Learning ecology": "Mediator",
    "Language load": "Mediator",
    "Epistemic stance": "Mediator",
    "AI understanding": "AI",
    "AI confidence": "AI",
}

def build_network_edges(pc_obj, adjust_method="fdr_bh", alpha=0.05, min_abs_r=0.0):
    tbl = partial_corr_table(pc_obj, upper_only=True).copy()

    reject, p_adj, _, _ = multipletests(tbl["p_value"], alpha=alpha, method=adjust_method)
    tbl["p_adj"] = p_adj
    tbl["sig"] = reject

    edges = tbl[
        (tbl["sig"]) & (tbl["partial_r"].abs() >= min_abs_r)
    ].copy()

    edges["weight"] = edges["partial_r"].abs()
    edges["sign"] = np.where(edges["partial_r"] >= 0, "positive", "negative")

    return edges.sort_values("weight", ascending=False).reset_index(drop=True)

def build_network_graph(edges, node_groups=None):
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(
            row["var1"],
            row["var2"],
            partial_r=row["partial_r"],
            weight=row["weight"],
            sign=row["sign"],
            p_adj=row["p_adj"]
        )

    if node_groups is not None:
        for node, group in node_groups.items():
            if node in G.nodes:
                G.nodes[node]["group"] = group

    return G


def network_centrality_table(G):
    if len(G.nodes) == 0:
        return pd.DataFrame(columns=["node", "degree", "strength", "betweenness"])

    degree = dict(G.degree())
    strength = {
        n: sum(abs(G[n][nbr]["partial_r"]) for nbr in G.neighbors(n))
        for n in G.nodes()
    }
    betweenness = nx.betweenness_centrality(G, weight="weight")

    out = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [degree[n] for n in G.nodes()],
        "strength": [strength[n] for n in G.nodes()],
        "betweenness": [betweenness[n] for n in G.nodes()],
    })

    return out.sort_values(["strength", "degree"], ascending=False).reset_index(drop=True)


def network_adjacency_matrix(G):
    nodes = list(G.nodes())
    mat = pd.DataFrame(0.0, index=nodes, columns=nodes)

    for u, v, data in G.edges(data=True):
        mat.loc[u, v] = data["partial_r"]
        mat.loc[v, u] = data["partial_r"]

    return mat





# =============================================================================
#  Visualization
# =============================================================================

def plot_partial_corr_heatmap(pc_obj, adjust_method="fdr_bh", alpha=0.05, sig_only=False, figsize=(10, 8)):
    est = pc_obj["estimate"].copy()
    tbl = partial_corr_table(pc_obj, upper_only=False)
    tbl = adjust_partial_corr_p(tbl, method=adjust_method, alpha=alpha)

    if sig_only:
        keep = tbl[tbl["sig"]][["var1", "var2"]].copy()
        keep["flag"] = 1

        flag_mat = pd.DataFrame(0, index=est.index, columns=est.columns)
        for _, row in keep.iterrows():
            flag_mat.loc[row["var1"], row["var2"]] = 1
            flag_mat.loc[row["var2"], row["var1"]] = 1

        for i in est.index:
            flag_mat.loc[i, i] = 1

        est = est.where(flag_mat == 1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        est,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title(
        f"Partial correlation matrix"
        + (f" ({adjust_method}, significant only)" if sig_only else "")
    )
    plt.tight_layout()
    plt.show()


def plot_network_graph(G, layout="spring", figsize=(10, 8), seed=2026):
    if len(G.nodes) == 0:
        print("No edges in graph.")
        return

    plt.figure(figsize=figsize)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=1.0)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed, k=1.0)

    edge_colors = [
        "#1f77b4" if G[u][v]["sign"] == "positive" else "#d62728"
        for u, v in G.edges()
    ]
    edge_widths = [
        1.5 + 6 * G[u][v]["weight"]
        for u, v in G.edges()
    ]

    # node colors by group if available
    default_color = "#cccccc"
    group_color_map = {
        "SES": "#f4a261",
        "Mediator": "#2a9d8f",
        "AI": "#e9c46a",
    }
    node_colors = []
    for n in G.nodes():
        grp = G.nodes[n].get("group", None)
        node_colors.append(group_color_map.get(grp, default_color))

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=3000,
        edgecolors="black",
        linewidths=1.5
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=20
    )

    plt.title("Partial-correlation network")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

