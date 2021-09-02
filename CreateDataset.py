from random import randint
from itertools import permutations
from csv import writer
from scipy.spatial import distance_matrix

import numpy as np
import os

""" Read the Coordinates and calculate the Distance Matrix from a file.tsp
"""
def get_coordinates_from_file(FOLDER_NAME, file_name):
    points = {}
    with open(f'{FOLDER_NAME}/{file_name}.tsp') as tsp_file:
        for line in tsp_file.readlines():
            try:
                ix, x, y = line.strip().split(' ')
                points[int(ix)] = (float(x), float(y))
            except ValueError:
                pass
    coords = np.array(list(list(i) for i in points.values()))

    dist_mat = distance_matrix(coords, coords)
    
    return coords, dist_mat

""" Read the Tour and the Tour Length from a file.tour
"""
def get_solution_from_file(FOLDER_NAME, file_name):
    tour_length = 0
    tour = []
    with open(f'{FOLDER_NAME}/{file_name}.tour') as tour_file:
        for idx, line in enumerate(tour_file.readlines()):
            try:
                if idx == 1:
                    for x in line.split(): 
                        if x.isdigit():
                            tour_length = int(x)
                step = line.strip()
                if step.isdigit():
                    tour.append(int(step))

            except ValueError:
                pass

    return tour_length, tour

def create_csv_file(FOLDER_NAME, file_name):
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    else:
        if os.path.exists(f'{FOLDER_NAME}/{file_name}.csv'):
            os.remove(f'{FOLDER_NAME}/{file_name}.csv')
    columns = [ 'A_x', 'A_y', 'B_x', 'B_y', 
                'C_x', 'C_y', 'D_x', 'D_y', 
                'E_x', 'E_y', 'AB', 'BC', 
                'CD', 'DE', 'Length', 'Good' ]
    with open(f'{FOLDER_NAME}/{file_name}.csv', 'w', newline='') as csv_file:
        # creating a csv writer object 
        csvwriter = writer(csv_file)
        # writing the fields 
        csvwriter.writerow(columns)

def add_record_to_csv(FOLDER_NAME, file_name, record):
    with open(f'{FOLDER_NAME}/{file_name}.csv', 'a', newline='') as csv_file:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(csv_file)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(record)

""" Create the Dataset
"""
def create_dataset(TSP_FOLDER, DATASET_FOLDER, dataset_name, N_RECORDS):
    create_csv_file(DATASET_FOLDER, dataset_name)
    for i in range(1, 101):
        file_name = f'problem_{i}'
        coords, distance_matrix = get_coordinates_from_file(TSP_FOLDER, file_name)
        tour_length, tour = get_solution_from_file(TSP_FOLDER, file_name)
        print(f'Problem: {file_name} \nTour Length: {tour_length} \nTOUR: {tour}')
        
        sorted_errors = get_sorted_errors(tour, tour_length, distance_matrix)
        print(f'SORTED ERRORS: {sorted_errors}\n\n')
        
        worst_indexes = [ x for x in list(sorted_errors)[:5] ]
        best_indexes = [ x for x in list(sorted_errors)[-5:] ] 
        #print(f'Best Indexes: {best_indexes} \nWorst Indexes: {worst_indexes}')
        
        best_sequences_list = get_best_sequences(tour)
        good_sequences_list = get_other_sequences(tour, best_indexes)
        bad_sequences_list = get_other_sequences(tour, worst_indexes)
        worst_sequences = get_worst_sequences(tour, worst_indexes, distance_matrix)
        
        for seq in best_sequences_list:
            record = create_record(seq, coords, distance_matrix, 1)
            add_record_to_csv(DATASET_FOLDER, dataset_name, record)

        for seq in good_sequences_list:
            record = create_record(seq, coords, distance_matrix, 1)
            add_record_to_csv(DATASET_FOLDER, dataset_name, record)
        
        for seq in bad_sequences_list:
            record = create_record(seq, coords, distance_matrix, 0)
            add_record_to_csv(DATASET_FOLDER, dataset_name, record)
        
        for seq in worst_sequences:
            record = create_record(seq, coords, distance_matrix, 0)
            add_record_to_csv(DATASET_FOLDER, dataset_name, record)

    """ TODO 
                (1)
                - Serializzare il dataset
                    - coordinate
                    - distanze tra nodi (archi)
                    - lunghezza percorso

                (2)
                - Generalizzare creazione del dataset per N_RECORD 
                - Utilizzare libreria generica per ottenere algoritmo di ML
                - Testare algoritmo ed eventualmente ampliare il dataset aumentando N_RECORD, sequenze
                    salvate oppure generando sequenze casuali

                (3)
                - Testare algoritmo di ML con nuovi file.tsp e file.tour generati casualmente

    """
    ...

def create_record(sequence, coords, distance_matrix, output):
    record = []
    for node in sequence:
        record.append(coords[node-1][0])
        record.append(coords[node-1][1])
    for subtour in zip(sequence, sequence[1:]):
        record.append(get_sequence_length(subtour, distance_matrix))
    record.append(get_sequence_length(sequence, distance_matrix))
    record.append(output)
    return record

def get_best_sequences(best_tour, n=5):
    random_indexes_list = []
    sequences_list = []
    while len(sequences_list) < n:
        matching_node = False
        random_node = randint(0,100)
        sequence = get_sequence(best_tour, random_node)
        #print(f'Node: {random_node}, Sequence: {sequence}')
                
        # Check if the generated node already belongs to a 
        for node in sequence:
            for seq in sequences_list:
                if node in seq:
                    matching_node = True
                    break
            if matching_node: 
                break  
                
        if not matching_node:
            random_indexes_list.append(random_node)
            sequences_list.append(sequence)
    
    return sequences_list

def get_other_sequences(tour, indexes_list):
    sequences_list = []
    for i in range(5):
        new_tour = tour.copy()
        # Exchange the i-node with its next one
        new_tour[indexes_list[i]] = tour[(indexes_list[i]+1) % len(tour)]
        new_tour[(indexes_list[i]+1) % len(tour)] = tour[indexes_list[i]]
        new_sequence = get_sequence(new_tour, indexes_list[i])
        #print(f'Index: {indexes_list[i]}, New Sequence: {new_sequence}')
        sequences_list.append(new_sequence)
    return sequences_list

def get_worst_sequences(tour, worst_indexes, distance_matrix):
    # [[seq1], [seq2], ..., [seq5]]
    error_dict = {}
    worst_sequences = []
    for index in worst_indexes:
        sequence = get_sequence(tour, index)
        sequence_length = get_sequence_length(sequence, distance_matrix)
        permutation_tuple = list(permutations(node for node in sequence[1:-1]))
        permutation_list = [list(i) for i in permutation_tuple]
        permutation_list.pop(0)

        for idx, bad_seq in enumerate(permutation_list):
            bad_seq.insert(0, sequence[0])
            bad_seq.insert(len(sequence), sequence[-1])
            bad_seq_length = get_sequence_length(bad_seq, distance_matrix)
            error = get_tour_error(sequence_length, bad_seq_length)
            error_dict[idx] = error
            print(f'ORIGINAL SEQUENCE: {sequence} -> BAD SEQUENCE: {bad_seq}')
            new_tour = create_new_tour(tour, index, bad_seq)
            new_tour_error = get_tour_error(get_tour_length(tour, distance_matrix), get_tour_length(new_tour, distance_matrix))
            print(f'ERROR: {new_tour_error}\n')

        print(permutation_list)
        error_dict_sorted = {k: v for k, v in sorted(error_dict.items(), key=lambda item: item[1], reverse = True)}
        worst_score_index = list(error_dict_sorted.keys())[0]
        worst_sequences.append(permutation_list[worst_score_index])
        print(f'{error_dict_sorted}\n\n\n')

    return worst_sequences

def create_new_tour(tour, index, sequence):
    #print(f'TOUR: {tour}')
    print(f'Index: {index}, Sequence to Replace: {sequence}')
    new_tour = tour.copy()
    new_tour[index] = sequence[2]
    new_tour[(index-1)%len(tour)] = sequence[1]
    new_tour[(index+1)%len(tour)] = sequence[3]
    #print(f'New Tour: {new_tour}')
    return new_tour

""" Return a Sequence of n nodes 

    Return a sequence of n nodes in the given tour.
        - tour : the array containing the nodes ordered by visit
        - candidate : the index of the node to take inside the sequence
    Default value for n is 5. A sequence of 5 nodes is returned. If n is even, the first
    (n/2) nodes before the candidate are taken with the last (n/2) following it.
    Otherwise, the first (n/2)-1 nodes before the candidate are taken with the last (n/2) following it.
"""
def get_sequence(tour, candidate, n=5):
    if n % 2 == 0:
        return [tour[i%len(tour)] for i in range(candidate-(n//2)+1, candidate+(n//2)+1)]
    else:
        return [tour[i%len(tour)] for i in range(candidate-(n//2), candidate+(n//2)+1)]

    
"""
    Return the errors of the new tours calculated exchanging each node of the tour by one position.
        - tour : the array containing the nodes ordered by visit
        - tour_length : the length of the tour to compare
        - distance_matrix : the matrix containing the distances of each node
"""
def get_sorted_errors(tour, tour_length, distance_matrix):
    error_dict = {}
    #print(f'ORIGINAL TOUR: {tour}')
    for i in range(0, 100):
        # Create a new solution starting from the best one
        new_tour = tour.copy()
        # Exchange the i-node with its next one
        new_tour[i] = tour[(i+1) % len(tour)]
        new_tour[(i+1) % len(tour)] = tour[i]
        # Calculate the new tour length
        new_tour_length = get_tour_length(new_tour, distance_matrix)
        # Calculate the error between the new tour and the best known
        error = get_tour_error(tour_length, new_tour_length)
        error_dict[i] = error
    error_dict_sorted = {k: v for k, v in sorted(error_dict.items(), key=lambda item: item[1], reverse = True)}
    return error_dict_sorted

"""
    Return the tour length
        - tour : the array containing the nodes ordered by visit
        - distance_matrix : the matrix containing the distances of each node
"""
def get_tour_length(tour, distance_matrix):
    total_distance = 0.0
    for idx, node in enumerate(tour):
        total_distance += round(distance_matrix[node-1][tour[(idx+1) % len(tour)]-1])
    return int(total_distance)

"""
    Return the sequence length
        - tour : the array containing the nodes ordered by visit
        - distance_matrix : the matrix containing the distances of each node
"""
def get_sequence_length(sequence, distance_matrix):
    total_distance = 0.0
    for idx, node in enumerate(sequence):
        if node != sequence[-1]:
            total_distance += round(distance_matrix[node-1][sequence[(idx+1)]-1])
    return int(total_distance)

"""
    Return the error between the given tour and the best known
        - best_tour_length : the length of the best tour known
        - tour_length : the length of the tour to compare
"""
def get_tour_error(best_tour_length, tour_length):
    return round(((tour_length - best_tour_length)/tour_length)*100, 2)

def main():
    TSP_FOLDER = "./TSPRandomGraph"
    DATASET_FOLDER = './Dataset'
    dataset_name = 'Dataset'
    SEQUENCE_DIMENSION = 5
    N_RECORDS = 20
    create_dataset(TSP_FOLDER, DATASET_FOLDER, dataset_name, N_RECORDS)
    # s = [75, 98, 1, 96, 22]
    # create_new_tour(tour, 0, s)


if __name__ == "__main__":
    main()    