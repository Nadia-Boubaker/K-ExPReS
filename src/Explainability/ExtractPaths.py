import random
import networkx as nx
import matplotlib.pyplot as plt

# =========================================================
# 1. Create filtered subgraph (no changes here)
# =========================================================
def create_filtered_user_course_topic_subgraph(G, user_node, valid_courses, recommended_course_node):
    nodes_in_subgraph = {user_node}  # Start with the user node
    edges_in_subgraph = []
    relevant_topics = set()

    # Identify knowledge topics linked to the recommended course
    recommended_topics = set(topic for topic in G.neighbors(recommended_course_node) if "cso.kmi.open.ac.uk/topics" in topic)

    # Add edges between the user and valid courses, and filter topics
    for course in valid_courses:
        if G.has_edge(user_node, course):
            nodes_in_subgraph.add(course)
            edge_data = G.get_edge_data(user_node, course)
            edges_in_subgraph.append((user_node, course, edge_data))
            # Add both 'hasKnowledgeTopic' and 'isKnowledgeTopicOf' edges
            for topic in G.neighbors(course):
                if topic in recommended_topics:
                    relevant_topics.add(topic)
                    nodes_in_subgraph.add(topic)
                    # Add 'hasKnowledgeTopic' edge
                    if G.has_edge(course, topic):
                        edge_data = G.get_edge_data(course, topic)
                        edges_in_subgraph.append((course, topic, edge_data))
                    # Add 'isKnowledgeTopicOf' edge
                    if G.has_edge(topic, course):
                        edge_data = G.get_edge_data(topic, course)
                        edges_in_subgraph.append((topic, course, edge_data))

    # Add the recommended course and its knowledge topics
    nodes_in_subgraph.add(recommended_course_node)
    for topic in recommended_topics:
        nodes_in_subgraph.add(topic)
        # Add 'hasKnowledgeTopic' edge
        if G.has_edge(recommended_course_node, topic):
            edge_data = G.get_edge_data(recommended_course_node, topic)
            edges_in_subgraph.append((recommended_course_node, topic, edge_data))
        # Add 'isKnowledgeTopicOf' edge
        if G.has_edge(topic, recommended_course_node):
            edge_data = G.get_edge_data(topic, recommended_course_node)
            edges_in_subgraph.append((topic, recommended_course_node, edge_data))

    # Create a subgraph with the selected nodes and edges
    subgraph = nx.DiGraph()
    subgraph.add_edges_from((u, v, d) for u, v, d in edges_in_subgraph)

    return subgraph


# =========================================================
# 2. Extract exactly 3 paths (unique lengths prioritized)
# =========================================================
def extract_three_paths(subgraph, user_node, recommended_course_node, max_length=7):
    """
    Extract exactly 3 paths:
      - Prefer paths of different lengths
      - If <3 unique lengths exist, fill the rest with additional paths
    """

    # Step 1: Get all simple paths
    all_paths = list(nx.all_simple_paths(
        subgraph,
        source=user_node,
        target=recommended_course_node,
        cutoff=max_length
    ))

    if not all_paths:
        print(f"⚠️ No paths found between {user_node} and {recommended_course_node}")
        return []

    # Step 2: Group paths by their length (# of edges)
    paths_by_length = {}
    for path in all_paths:
        length = len(path) - 1  # number of edges
        if length not in paths_by_length:
            paths_by_length[length] = []
        paths_by_length[length].append(path)

    # Step 3: Select one path per unique length (first pass)
    selected_paths = []
    used_lengths = set()

    for length in sorted(paths_by_length.keys()):
        if len(selected_paths) >= 3:
            break
        chosen_path = random.choice(paths_by_length[length])
        selected_paths.append(chosen_path)
        used_lengths.add(length)

    # Step 4: If we still need more paths, fill from any length
    if len(selected_paths) < 3:
        remaining_paths = [p for paths in paths_by_length.values() for p in paths]
        # Remove already chosen paths to avoid duplicates
        remaining_paths = [p for p in remaining_paths if p not in selected_paths]

        while len(selected_paths) < 3 and remaining_paths:
            selected_paths.append(random.choice(remaining_paths))
            remaining_paths.remove(selected_paths[-1])

    # Step 5: Convert to detailed format with relations
    detailed_paths = []
    for path in selected_paths:
        path_with_relations = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            predicate_uri = subgraph[source][target].get('predicate', 'UNKNOWN')
            predicate_label = predicate_uri.split('/')[-1]  # simplify relation URI
            path_with_relations.append((source, predicate_label))
        path_with_relations.append((path[-1], None))  # final node
        detailed_paths.append(path_with_relations)

    return detailed_paths


# =========================================================
# 3. Example usage with your subgraph
# =========================================================
# Replace with actual data
user_node = user_uri
valid_courses = set(courses_with_same_topics)
recommended_course_node = recommended_course_uri

# Build subgraph
subgraph = create_filtered_user_course_topic_subgraph(G, user_node, valid_courses, recommended_course_node)

# Extract exactly 3 paths
detailed_paths = extract_three_paths(
    subgraph=subgraph,
    user_node=user_node,
    recommended_course_node=recommended_course_node,
    max_length=7
)

# Display results
print(f"\nDisplaying {len(detailed_paths)} paths:\n")
for i, path in enumerate(detailed_paths, 1):
    path_str = ""
    for node, relation in path:
        node_label = node.split('/')[-1]  # Simplify node URI
        path_str += node_label
        if relation:
            path_str += f" --[{relation}]--> "
    print(f"Path {i}: {path_str}")


