import csv
from datetime import datetime, timedelta
from collections import defaultdict
from igraph import Graph
import networkx as nx
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
import random
import os

#Generate a random seed
random.seed(42)
np.random.seed(42)

# Load the diagnosis data 
diagnosis_data = defaultdict(list)
with open('normalizedDiagnosis.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        mrn = row[0]
        perform_date = datetime.strptime(row[11], '%m/%d/%y')
        icd_code = row[15]
        diagnosis_data[mrn].append((perform_date, icd_code))

# Load the medication 
medication_data = defaultdict(list)
with open('normalizedMedication.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        mrn = row[0]
        perform_date = datetime.strptime(row[7], '%m/%d/%y')
        medication_group = row[9]
        medication_data[mrn].append((perform_date, medication_group))

# Load the lab tests data 
lab_data = defaultdict(list)
with open('normalizedLab.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        mrn = row[0]
        lab_name = row[7]
        perform_date = datetime.strptime(row[10], '%m/%d/%y')
        normal_yn = row[12]
        lab_data[mrn].append((perform_date, lab_name, normal_yn))

# Get unique MRNs from all data sources
all_mrns = sorted(set(diagnosis_data.keys()) | set(medication_data.keys()) | set(lab_data.keys()))

for selected_mrn in all_mrns:
    # Create a new graph for the MRN
    patient_graph = Graph()
    vertex_indices = {}
    merged_vertex_indices = {}

    def get_merged_vertex_index(vertex_name, occurrences, perform_date):
        key = (vertex_name, perform_date)  # Create a tuple with vertex name and perform date

        if key in merged_vertex_indices:
            # Get the merged index if the vertex with the same name and date already exists
            merged_vertex_index = merged_vertex_indices[key]
            merged_vertex = patient_graph.vs[merged_vertex_index]
            merged_vertex["occurrences"] += occurrences
            return merged_vertex_index
        else:
            # Add a new vertex and store its index in the merged_vertex_indices dictionary
            new_vertex_index = patient_graph.add_vertex(name=vertex_name, occurrences=occurrences, perform_date=perform_date).index
            merged_vertex_indices[key] = new_vertex_index
            return new_vertex_index

    # Add nodes for diagnosis
    diagnoses = diagnosis_data[selected_mrn]
    for perform_date, icd_code in diagnoses:
        vertex_name = f"diagnosis:{icd_code}"
        get_merged_vertex_index(vertex_name, 1, perform_date)

    # Add nodes for medication groups
    medications = medication_data[selected_mrn]
    for perform_date, medication_group in medications:
        vertex_name = f"medication:{medication_group}"
        get_merged_vertex_index(vertex_name, 1, perform_date)

    # Add nodes for lab tests
    labs = lab_data[selected_mrn]
    for perform_date, lab_name, normal_yn in labs:
        if normal_yn == 'High':
            node_name = f'{lab_name}+high'
        elif normal_yn == 'Low':
            node_name = f'{lab_name}+low'
        else:
            node_name = f'{lab_name}+abnormal'
        vertex_name = f"lab:{node_name}"
        get_merged_vertex_index(vertex_name, 1, perform_date)

    # Calculate co-occurrences and update edge weights
    co_occurrence_counts = defaultdict(int)
    existing_edges = set()

    for i, node1 in enumerate(patient_graph.vs):
        for j, node2 in enumerate(patient_graph.vs[i+1:], start=i+1):
            perform_date1 = node1["perform_date"]
            perform_date2 = node2["perform_date"]
            if abs(perform_date1 - perform_date2) <= timedelta(days=180):
                edge_key = (min(node1.index, node2.index), max(node1.index, node2.index))
                edge_indices = (node1.index, node2.index)
                if edge_indices not in existing_edges:
                    existing_edges.add(edge_indices)
                co_occurrence_counts[edge_key] += 1

    # Merge duplicate edges and update their weights
    merged_patient_graph = Graph()
    merged_vertex_indices = {}
    merged_vertex_index = 0

    for (source_index, target_index), co_occurrences in co_occurrence_counts.items():
        source_name = patient_graph.vs[source_index]['name']
        target_name = patient_graph.vs[target_index]['name']
        
        if source_name not in merged_vertex_indices:
            merged_vertex_indices[source_name] = merged_vertex_index
            merged_patient_graph.add_vertex(name=source_name, occurrences=0, perform_date=None)
            merged_vertex_index += 1
        
        if target_name not in merged_vertex_indices:
            merged_vertex_indices[target_name] = merged_vertex_index
            merged_patient_graph.add_vertex(name=target_name, occurrences=0, perform_date=None)
            merged_vertex_index += 1
        
        merged_source_index = merged_vertex_indices[source_name]
        merged_target_index = merged_vertex_indices[target_name]
        
        merged_edge_index = merged_patient_graph.get_eid(merged_source_index, merged_target_index, directed=False, error=False)
        if merged_edge_index == -1:
            merged_patient_graph.add_edge(merged_source_index, merged_target_index, weight=co_occurrences)
        else:
            merged_patient_graph.es[merged_edge_index]['weight'] += co_occurrences

    # Convert the graph from igraph to networkx
    G = nx.Graph()
    for vertex in merged_patient_graph.vs:
        G.add_node(vertex.index, name=vertex['name'], occurrences=vertex['occurrences'], perform_date=vertex['perform_date'])
    for edge in merged_patient_graph.es:
        G.add_edge(edge.source, edge.target, weight=edge['weight'])

    # Create a Node2Vec object
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, p=1.0, q=1.0, weight_key='weight',seed=42)

    # Generate the random walks
    walks = node2vec.walks

    # Learn the node embeddings
    model = node2vec.fit(window=10, min_count=1, epochs=10)

    # Retrieve the embeddings for all nodes
    embeddings = {node: model.wv[node] for node in G.nodes()}

    # Convert the embeddings to a list
    embedding_list = list(embeddings.values())

    # Set the range of possible number of clusters
    min_clusters = 2
    max_clusters = 10

    # Initialize variables to store the optimal number of clusters and its corresponding silhouette score
    optimal_num_clusters = None
    max_silhouette_score = -1

    # Iterate over the range of possible number of clusters
    for num_clusters in range(min_clusters, max_clusters + 1):
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_list)

        # Calculate the silhouette score
        silhouette_avg = silhouette_score(embedding_list, cluster_labels)

        # Update the optimal number of clusters if the current silhouette score is higher
        if silhouette_avg > max_silhouette_score:
            optimal_num_clusters = num_clusters
            max_silhouette_score = silhouette_avg

    # Set the number of clusters as the optimal number of clusters
    num_clusters = optimal_num_clusters

    # Perform K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding_list)

    # Create a mapping dictionary for node names and cluster labels
    node_cluster_mapping = {node: label for node, label in zip(embeddings.keys(), cluster_labels)}

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_

    centroid_names = []
    for centroid in centroids:
        centroid_node = None
        min_distance = float('inf')
        for node in merged_patient_graph.vs:
            node_name = node['name']
            if node_name.startswith("diagnosis") or node_name.startswith("medication") or node_name.startswith("lab"):
                node_embedding = embeddings[node.index]
                distance = np.linalg.norm(node_embedding - centroid)
                if distance < min_distance:
                    min_distance = distance
                    centroid_node = node_name
        centroid_names.append(centroid_node)


    # Group nodes based on their cluster labels
    clustered_nodes = defaultdict(list)
    for node, label in node_cluster_mapping.items():
        clustered_nodes[label].append(node)

    # Sort the clustered_nodes dictionary based on cluster labels
    clustered_nodes = dict(sorted(clustered_nodes.items()))

    #Create the output folder
    output_folder = "output_test"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Specify the file path to save the output
    output_file = os.path.join(output_folder,f"test_centroid_vector_{selected_mrn}.txt")

    # Open the output file in write mode
    with open(output_file, "w") as file:
        # Write the combined centroid vector to the file
        for cluster_label, nodes_in_cluster in clustered_nodes.items():
            node_names = [merged_patient_graph.vs[node]['name'] for node in nodes_in_cluster]
            file.write(",".join(node_names) + "\n")

        # Write the cluster centroids to the file
        centroid_names = [centroid for centroid in centroid_names]
        file.write(",".join(centroid_names))

