import networkx as nx
import pandas as pd

# Example weights 
beta = 0.4     # Simplicity
gamma = 0.3   # Popularity
delta = 0.3   # Diversity

# Step 1: Compute PageRank of the subgraph
pagerank_scores = nx.pagerank(subgraph)

# Step 2: Compute path scores
path_scores = []

for path_with_relations in diverse_paths:
    # Extract nodes only
    nodes = [node for node, _ in path_with_relations]

    # S1: Simplicity = 1 / path length
    simplicity = 1 / (len(nodes) - 1) if len(nodes) > 1 else 0

    # Popularity = average PageRank of intermediate nodes
    popularity = sum(pagerank_scores.get(n, 0) for n in nodes) / len(nodes)

    path_scores.append({
        "Path": path_with_relations,
        "Nodes": set(nodes),
        "Simplicity (1/len)": simplicity,
        "Popularity (PageRank)": popularity
    })

# Step 3: Compute Diversity (1 - Jaccard similarity with other paths)
for i in range(len(path_scores)):
    similarities = []
    nodes_i = path_scores[i]["Nodes"]
    for j in range(len(path_scores)):
        if i != j:
            nodes_j = path_scores[j]["Nodes"]
            intersection = len(nodes_i & nodes_j)
            union = len(nodes_i | nodes_j)
            jaccard = intersection / union if union != 0 else 0
            similarities.append(jaccard)
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    diversity = 1 - avg_similarity
    path_scores[i]["Diversity (Jaccard)"] = diversity

# Step 4: Compute final composite score
table_data = []
for i, score in enumerate(path_scores, 1):
    composite_score = (
        beta * score["Simplicity (1/len)"] +
        gamma * score["Popularity (PageRank)"] +
        delta * score["Diversity (Jaccard)"]
    )
    table_data.append({
        "Path #": i,
        "Simplicity (P)": round(score["Simplicity (1/len)"], 4),
        "Popularity (P)": round(score["Popularity (PageRank)"], 4),
        "Diversity (P)": round(score["Diversity (Jaccard)"], 4),
        "Final_Score(P)": round(composite_score, 4)
    })

# Step 5: Display DataFrame
df_scores = pd.DataFrame(table_data).sort_values(by="Final_Score(P)", ascending=False).reset_index(drop=True)
print(df_scores[["Path #", "Simplicity (P)", "Popularity (P)", "Diversity (P)", "Final_Score(P)"]].to_string(index=False))
ranking_algo.py
Displaying ranking_algo.py.