import os
import csv
import pandas as pd

def find_overlapping_centroids(mrn_txt_file, centroid_dir):
    with open(mrn_txt_file, 'r') as file:
        mrn_lines = file.readlines()

    centroids_dict = {}
    overlapping_centroids = {}
    overlapping_nodes = {}

    for line in mrn_lines:
        line = line.strip()
        if line.startswith("Centroid:"):
            centroid = line.split(":", 1)[-1].strip()
        elif line.startswith("MRN:"):
            mrn = line.split(":")[-1].strip()
            centroids_dict[mrn] = centroid

    for mrn, centroid in centroids_dict.items():
        centroid_vector_file = os.path.join(centroid_dir, f"centroid_vector_{mrn}.txt")

        if os.path.exists(centroid_vector_file):
            with open(centroid_vector_file, 'r') as cv_file:
                cv_lines = cv_file.readlines()

            centroids_in_file = cv_lines[-2].strip().split(',')
            overlapping_centroid_index = None

            for idx, c in enumerate(centroids_in_file):
                if c == centroid:
                    overlapping_centroid_index = idx
                    break

            if overlapping_centroid_index is not None:
                overlapping_centroids[mrn] = centroid

                nodes_line = cv_lines[overlapping_centroid_index].strip()
                
                cluster_nodes = []
                node_parts = []
                for node in nodes_line.split(','):
                    node_parts.append(node.strip())
                if node_parts:
                    cluster_nodes.append(' '.join(node_parts))
                
                overlapping_nodes[centroid] = cluster_nodes

    return overlapping_centroids, overlapping_nodes

def combine_nodes_into_vector(overlapping_nodes):
    all_nodes = []
    for centroid, nodes in overlapping_nodes.items():
        all_nodes.extend(nodes)
    return all_nodes

def format_node(node):
    node = node.replace("[", "").replace("]", "")
    return node

def output_vector_to_txt(all_nodes, output_file):
    with open(output_file, 'w') as file:
        for node in all_nodes:
            formatted_node = format_node(node)
            file.write(f"{formatted_node}\n")

def remove_duplicates_from_vector(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    unique_lines = list(dict.fromkeys(lines))

    with open(output_file, 'w') as file:
        file.writelines(unique_lines)

centroid_dir = 'training_data'
mrn_txt_file = 'training_centroids_and_phenotypes.txt'

overlapping_info = find_overlapping_centroids(mrn_txt_file, centroid_dir)
all_nodes = combine_nodes_into_vector(overlapping_info)

output_file = 'phenotype_vector.txt'
output_vector_to_txt(all_nodes, output_file)

def get_stroke_value(mrn):
    file_path = os.path.join("testing_data", f"centroid_vector_{mrn}.txt")
    with open(file_path, "r") as f:
        lines = f.readlines()
    stroke_value = int(lines[-1].strip())
    return stroke_value

mrn_list_train = []
with open("training_list.txt", "r") as f:
    for line in f:
        mrn_list_train.append(line.strip())

data_train = []
for mrn in mrn_list_train:
    stroke_value = get_stroke_value(mrn)
    data_train.append((mrn, stroke_value))

node_names_train = []
with open("phenotype_vector.txt", "r") as f:
    node_names_train = [line.strip() for line in f]

csv_file_path_train = "training_patient_representation.csv"
with open(csv_file_path_train, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["mrn", "stroke"] + node_names_train)
    
    for mrn, stroke_value in data_train:
        writer.writerow([mrn, stroke_value] + [""] * len(node_names_train))

# Finding overlapping phenotypes for testing patients
mrn_list_test = []
with open("testing_list.txt", "r") as f:
    for line in f:
        mrn_list_test.append(line.strip())

data_test = []
overlapping_info_test = {}  # Dictionary to store overlapping info for testing patients

for mrn in mrn_list_test:
    stroke_value = get_stroke_value(mrn)
    data_test.append((mrn, stroke_value))
    overlapping_info_test[mrn] = find_overlapping_centroids(f"centroid_vector_{mrn}.txt", centroid_dir)

node_names_test = []
with open("phenotype_vector.txt", "r") as f:
    node_names_test = [line.strip() for line in f]

csv_file_path_test = "testing_patient_representation.csv"
with open(csv_file_path_test, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["mrn", "stroke"] + node_names_test)
    
    for mrn, stroke_value in data_test:
        writer.writerow([mrn, stroke_value] + [""] * len(node_names_test))

def update_csv_with_txt_data(csv_file_path, folder_path):
    df = pd.read_csv(csv_file_path)

    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    for txt_file in txt_files:
        txt_file_path = os.path.join(folder_path, txt_file)
        mrn = txt_file.split('_')[2].split('.')[0]

        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            node_name = line.strip()
            if node_name in df.columns[2:]:
                df.loc[df['mrn'] == int(mrn), node_name] = 1

    df.to_csv(csv_file_path, index=False)

update_csv_with_txt_data(csv_file_path_train, 'present_phenotypes_training.csv')
update_csv_with_txt_data(csv_file_path_test, 'present_phenotypes_testing.csv')
