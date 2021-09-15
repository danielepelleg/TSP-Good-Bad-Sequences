from random import randint, shuffle
from itertools import permutations
from csv import writer
from scipy.spatial import distance_matrix
from string import ascii_uppercase

import numpy as np
import argparse
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

""" Create the CSV dataset file
"""
def create_csv_file(FOLDER_NAME, N_RECORDS, N_SEQUENCE, ERROR_TARGET):
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    else:
        if os.path.exists(f'{FOLDER_NAME}/Dataset_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCE}_ERROR{ERROR_TARGET}.csv'):
            os.remove(f'{FOLDER_NAME}/Dataset_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCE}_ERROR{ERROR_TARGET}.csv')
    alphabet = list(ascii_uppercase)
    columns = []
    
    for l in range(0, N_SEQUENCE):
        columns.append(f'{alphabet[l]}_x')
        columns.append(f'{alphabet[l]}_y')
    
    for l in range(0, N_SEQUENCE-1):
        columns.append(f'{alphabet[l]}{alphabet[l+1]}')
    
    columns.append('Length')
    columns.append('Good')

    with open(f'{FOLDER_NAME}/Dataset_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCE}_ERROR{ERROR_TARGET}.csv', 'w', newline='') as csv_file:
        # creating a csv writer object 
        csvwriter = writer(csv_file)
        # writing the fields 
        csvwriter.writerow(columns)

""" Add a record in the CSV dataset file
"""
def add_record_to_csv(FOLDER_NAME, N_RECORDS, N_SEQUENCE, ERROR_TARGET, record):
    with open(f'{FOLDER_NAME}/Dataset_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCE}_ERROR{ERROR_TARGET}.csv', 'a', newline='') as csv_file:
        # Pass this file object to csv.writer() and get a writer object
        writer_object = writer(csv_file)
    
        # Pass the list as an argument into the writerow()
        writer_object.writerow(record)

""" Create the Record to insert in the Dataset
"""
def create_record(sequence, coords, distance_matrix, output):
    record = []
    for node in sequence:
        record.append(coords[node-1][0])
        record.append(coords[node-1][1])
    for subtour in zip(sequence, sequence[1:]):
        record.append(get_sequence_length(subtour, distance_matrix))
    record.append(get_sequence_length(sequence, distance_matrix))
    if output is not None:
        record.append(output)
    return record

""" Create the Dataset
"""
def create_dataset(N_GRAPHS, TSP_FOLDER, DATASET_FOLDER, N_RECORDS, N_SEQUENCE, ERROR_TARGET):
    n_samples = N_RECORDS//4 
    create_csv_file(DATASET_FOLDER, N_RECORDS, N_SEQUENCE, ERROR_TARGET)
    for i in range(1, N_GRAPHS+1):
        file_name = f'problem_{i}'
        print(f'Problem: {file_name}\n')
        coords, distance_matrix = get_coordinates_from_file(TSP_FOLDER, file_name)
        tour_length, tour = get_solution_from_file(TSP_FOLDER, file_name)
        random_tour = get_random_tour(tour_length, distance_matrix)
        #print(f'Tour Length: {tour_length} \nTOUR: {tour}')
    
        sorted_errors, sequences_dict = get_sorted_errors(tour, tour_length, distance_matrix, N_SEQUENCE)
        #print(f'SORTED ERRORS: {sorted_errors}\n SEQUENCE DICTIONART: {sequences_dict}\n') # Performing one enxchange for node
        
        worst_indexes = [ x for x in list(sorted_errors)[:n_samples] ]
        best_indexes = [ x for x in list(sorted_errors) if sorted_errors[x] <= ERROR_TARGET]
        #print(f'Best Indexes: {best_indexes} \nWorst Indexes: {worst_indexes}')
        
        best_sequences_list = get_unique_sequences(tour, n_samples, N_SEQUENCE)
        print(f'Best Sequences: {best_sequences_list}\n')
        good_sequences_list = get_good_sequences(best_indexes, sequences_dict, n_samples)
        print(f'Good Sequences: {good_sequences_list}\n')
        bad_sequences_list = get_bad_sequences(tour, worst_indexes, distance_matrix, N_SEQUENCE, ERROR_TARGET)
        # Additional samples to add as a worst sequence in the dataset 
        # if not enough bad sequences are found
        additional_samples = n_samples - len(bad_sequences_list)
        print(f'Bad Sequences: {bad_sequences_list}\tAdditional Samples: {additional_samples}\n')
        worst_sequences_list = get_random_sequences(random_tour, n_samples + additional_samples, N_SEQUENCE)
        print(f'Worst Sequences: {worst_sequences_list}\n')
        
        for seq in best_sequences_list:
            record = create_record(seq, coords, distance_matrix, 1)
            add_record_to_csv(DATASET_FOLDER, N_RECORDS, N_SEQUENCE, ERROR_TARGET, record)

        for seq in good_sequences_list:
            record = create_record(seq, coords, distance_matrix, 1)
            add_record_to_csv(DATASET_FOLDER, N_RECORDS, N_SEQUENCE, ERROR_TARGET, record)
        
        for seq in bad_sequences_list:
            record = create_record(seq, coords, distance_matrix, 0)
            add_record_to_csv(DATASET_FOLDER, N_RECORDS, N_SEQUENCE, ERROR_TARGET, record)
        
        for seq in worst_sequences_list:
            record = create_record(seq, coords, distance_matrix, 0)
            add_record_to_csv(DATASET_FOLDER, N_RECORDS, N_SEQUENCE, ERROR_TARGET, record)

""" Create a Random Tour
"""
def get_random_tour(best_tour_length, dist_mat):
    TARGET_ERROR_RANDOM_TOUR = 15
    random_tour = [i for i in range(1,101)]
    valid = False
    while not valid:
        shuffle(random_tour)
        random_tour_length = get_tour_length(random_tour, dist_mat)
        error = get_tour_error(best_tour_length, random_tour_length)
        if error >= TARGET_ERROR_RANDOM_TOUR:
            valid = True
    return random_tour

""" Get Unique Sequences from a Tour

    Return a list of sequences from a tour known, choosing them so their nodes are unique.
        - tour : the tour known
        - n_samples : number of samples to insert in the sequences list
        - N_SEQUENCE : the number of nodes in the sequence 
"""
def get_unique_sequences(tour, n_samples, N_SEQUENCE):
    # The sequences are unique until the 70% of the nodes are taken
    TARGET = int(len(tour)*0.7)
    random_indexes_set = set()
    sequences_list = []
    random_node_index = randint(0,len(tour))
    counter = 1
    while len(sequences_list) < n_samples:
        if n_samples * N_SEQUENCE <= TARGET:
            if random_node_index not in random_indexes_set:
                random_indexes_set.add(random_node_index)
                # Add the N_SEQUENCE-1 neighbors of the random node chosen
                for i in range(-(N_SEQUENCE-1),N_SEQUENCE):
                    random_indexes_set.add((random_node_index+i)%len(tour))
                sequence = get_sequence(tour, random_node_index, N_SEQUENCE)
                sequences_list.append(sequence)
            random_node_index = randint(0,len(tour))
        else:
            # If the seqeuences list is empty
            if not sequences_list:
                sequence = get_sequence(tour, random_node_index, N_SEQUENCE)
                sequences_list.append(sequence)
            # Take the previous and the next sequence and add them to the list
            prev_sequence = get_sequence(tour, (random_node_index-(counter*N_SEQUENCE))%len(tour), N_SEQUENCE)
            next_sequence = get_sequence(tour, (random_node_index+(counter*N_SEQUENCE))%len(tour), N_SEQUENCE)
            sequences_list.append(prev_sequence)
            sequences_list.append(next_sequence)
            counter += 1
    # If the list contains one more sequence
    if len(sequences_list) > n_samples:
        # Delete one random between the last two inserted
        sequences_list.pop(randint(-2,0))
    return sequences_list

""" Get a Random Sequence from a Tour
    
    Return a list of sequences from a tour known, choosing them randomly.
        - tour : the tour known
        - n_samples : number of samples to insert in the sequences list
        - N_SEQUENCE : the number of nodes in the sequence
"""
def get_random_sequences(tour, n_samples, N_SEQUENCE):
    random_indexes_list = []
    sequences_list = []
    while len(sequences_list) < n_samples:
        random_node = randint(0,len(tour))
        if random_node in random_indexes_list:
            continue
        sequence = get_sequence(tour, random_node, N_SEQUENCE)
        #print(f'Node: {random_node}, Sequence: {sequence}')
        
        random_indexes_list.append(i for i in range((random_node-1)%len(tour), (random_node+2)%len(tour)))
        sequences_list.append(sequence)
    
    return sequences_list

""" Get the other Sequences obtained by doing some exchanges.

    Return the sequences obtained by by exchanging the nodes corresponding to the indexes in the list given.
    Each node it's exchanged with its one following.
        - tour : the array containing the nodes ordered by visit
        - indexes_list : the list containg the indexes of a tour
        - n_samples : number of samples to insert in the sequences list
        - N_SEQUENCE : the number of nodes in the sequence
"""
def get_good_sequences(best_indexes, sequences_dictionary, n_samples):
    sequences_list = []
    candidates = len(best_indexes)-1
    step = candidates/(n_samples-1)
    if step < 1.0:
        sequences_list.append(sequences_dictionary[i] for i in best_indexes)
    else:
        for i in range(n_samples):
            sequences_list.append(sequences_dictionary[best_indexes[round(i*step)]])
    return sequences_list

""" Get Bad Sequences on a Graph by doing some permutations.

    Return the bad sequences by doing all the permutations for each worst_index given. A bad sequence 
    is one which has a major error from the original one belonging to the tour. The sequence compared
    must have the same beginning node and the same ending node.
        - tour : the array containing the nodes ordered by visit
        - worst_index : the list containg the worst indexes of a tour
        - distance_matrix : the matrix containing the distances of each node
        - N_SEQUENCE : the number of nodes in the sequence
        - ERROR_TARGET : the error to use to choose the sequence to discard
"""
def get_bad_sequences(tour, worst_indexes, distance_matrix, N_SEQUENCE, ERROR_TARGET):
    error_dict = {}
    for index in worst_indexes:
        sequence = get_sequence(tour, index, N_SEQUENCE) # [n1, n2, n3, n4, n5]
        sequence_length = get_sequence_length(sequence, distance_matrix)
        # Permutate the nodes in the middle of the sequence [n1, n2, n3, n4, n5] -> [n2, n3, n4]
        permutation_tuple = list(permutations(node for node in sequence[1:-1]))
        permutation_list = [list(i) for i in permutation_tuple] # [ [n2, n3, n4], [n4, n2, n3], ...]
        # Remove the sequence known from the permutation list
        permutation_list.pop(0) # [[n4, n2, n3], ...]

        for bad_seq in permutation_list:
            # Add the two edge nodes to the sequence [n4, n2, n3] -> [n1, n4, n2, n3, n5]
            bad_seq.insert(0, sequence[0])
            bad_seq.insert(len(sequence), sequence[-1])
            bad_seq_length = get_sequence_length(bad_seq, distance_matrix)
            new_sequence_error = get_tour_error(sequence_length, bad_seq_length)
            #print(f'ORIGINAL SEQUENCE: {sequence} -> BAD SEQUENCE: {bad_seq}')
            new_tour = create_new_tour(tour, index, bad_seq)
            new_tour_error = get_tour_error(get_tour_length(tour, distance_matrix), get_tour_length(new_tour, distance_matrix))
            error_dict[tuple(bad_seq)] = new_tour_error # {([n1, n4, n2, n3, n5]): error, ...}
            #print(f'ERROR: {new_tour_error}\n')
    
    # Sort the Dictionary starting from the highest errors
    error_dict_sorted = {k: v for k, v in sorted(error_dict.items(), key=lambda item: item[1], reverse = True)}
    # print(f'Length Error Dictionary Sorted: {len(error_dict_sorted)}, Middle Value: {list(error_dict_sorted.values())[len(error_dict_sorted)//2]}')
    # print(error_dict_sorted)

    # Take as the smallest error the one in the middle of the dictionary
    smallest_error_index = len(error_dict_sorted)//2
    smallest_error = list(error_dict_sorted.values())[smallest_error_index]
    # Find the smallest error which is > ERROR_TARGET
    # Add the element to the list choosing them if they are suitable with the given target
    while(smallest_error > ERROR_TARGET):
        if smallest_error_index == len(error_dict_sorted)-1:
            break
        next_smallest_error = list(error_dict_sorted.values())[smallest_error_index+1]
        if next_smallest_error > ERROR_TARGET:
            smallest_error_index += 1
            smallest_error = next_smallest_error
        else: break
    while(smallest_error <= ERROR_TARGET):
        if smallest_error_index == 0: 
            return []
        smallest_error_index -= 1
        smallest_error = list(error_dict_sorted.values())[smallest_error_index]
    # Calculate the step to use to take the sequence to add in the dataset
    #(f'Smallest Error Index: {smallest_error_index}, Smallest Error: {smallest_error}')
    step = smallest_error_index/(len(worst_indexes)-1)
    #print(f'Step: {step}')
    worst_sequences = []
    worst_errors = []
    # If the number os samples to take is major than the errors available 
    # all the sequence available are taken to be added in the dataset
    if step < 1.0:
        while(smallest_error_index >= 0):
            worst_errors.append(list(error_dict_sorted.values())[smallest_error_index])
            worst_sequences.append(list(error_dict_sorted.keys())[smallest_error_index])
            smallest_error_index -= 1
    # Add a sequence increasing one step at time the index used to choose 
    else:
        for i in range(len(worst_indexes)):
            index_node = smallest_error_index-(round(i*step))
            worst_errors.append(list(error_dict_sorted.values())[index_node])
            worst_sequences.append(list(error_dict_sorted.keys())[index_node])
    
    #print(f'Worst Errors: {worst_errors}')
    #print(f'Worst Sequences: {worst_sequences}')
    return worst_sequences

""" Create a New Tour replacing a Sequence

    Return a new tour replacing a sequence in the given one.
        - tour : the array containing the nodes ordered by visit
        - index : the index of the node to take inside the sequence
        - N_SEQUENCE : the number of nodes in the sequence
"""
def create_new_tour(tour, index, sequence):
    #print(f'TOUR: {tour}')
    n_sequence = len(sequence)
    #print(f'Index: {index}, Sequence to Replace: {sequence}')
    new_tour = tour.copy()
    # Even Sequence
    if n_sequence % 2 == 0:
        for idx, i in enumerate(range(-(n_sequence//2)+1, (n_sequence//2)+1)):
            new_tour[(index+i)%len(tour)] = sequence[idx]
    # Odd Sequence
    else:
        for idx, i in enumerate(range(-(n_sequence//2), (n_sequence//2)+1)):
            new_tour[(index+i)%len(tour)] = sequence[idx]
    #print(f'New Tour: {new_tour}')

    return new_tour

""" Return a Sequence of n nodes 

    Return a sequence of n nodes in the given tour.
        - tour : the array containing the nodes ordered by visit
        - index : the index of the node to take inside the sequence
        - N_SEQUENCE : the number of nodes in the sequence
    Default value for n is 5. A sequence of 5 nodes is returned. If n is even, the first
    (n/2) nodes before the index are taken with the last (n/2) following it.
    Otherwise, the first (n/2)-1 nodes before the index are taken with the last (n/2) following it.
"""
def get_sequence(tour, index, N_SEQUENCE):
    if N_SEQUENCE % 2 == 0:
        return [tour[i%len(tour)] for i in range(index-(N_SEQUENCE//2)+1, index+(N_SEQUENCE//2)+1)]
    else:
        return [tour[i%len(tour)] for i in range(index-(N_SEQUENCE//2), index+(N_SEQUENCE//2)+1)]
    
"""
    Return the errors of the new tours calculated exchanging each node of the tour by one position.
        - tour : the array containing the nodes ordered by visit
        - tour_length : the length of the tour to compare
        - distance_matrix : the matrix containing the distances of each node
"""
def get_sorted_errors(tour, tour_length, distance_matrix, N_SEQUENCE):
    errors_dict = {} # {index: error}
    sequences_dict = {} # {index: [seq]}
    #print(f'ORIGINAL TOUR: {tour}')
    for i in range(0, len(tour)):
        # Create a new solution starting from the best one
        new_tour = tour.copy()
        # Exchange the i-node with its next one
        new_tour[i] = tour[(i+1) % len(tour)]
        new_tour[(i+1) % len(tour)] = tour[i]
        new_sequence = get_sequence(new_tour, i, N_SEQUENCE)
        # Calculate the new tour length
        new_tour_length = get_tour_length(new_tour, distance_matrix)
        # Calculate the error between the new tour and the best known
        error = get_tour_error(tour_length, new_tour_length)
        errors_dict[i] = error # {index: error, ...}
        sequences_dict[i] = new_sequence # {index: seq, ...}

    # Sort the Dictionary starting from the highest errors
    errors_dict_sorted = {k: v for k, v in sorted(errors_dict.items(), key=lambda item: item[1], reverse = True)}
    return errors_dict_sorted, sequences_dict

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
        # A sequence doesn't need to close the last node with the first one
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

""" Validate the Inputs
"""
def validate_input(N_NODES, N_SEQUENCE, N_RECORDS, ERROR_TARGET):
    try:
        assert N_RECORDS % 4 == 0 and 4 <= N_SEQUENCE <= 8 and 2 <= ERROR_TARGET <= 5 and ((N_RECORDS//4) * N_SEQUENCE) <= N_NODES
    except AssertionError:
        if (N_RECORDS % 4) != 0:
            print(f'Invalid Number of Records: {N_RECORDS}. Please insert Number of Records divisible by 4.')
        if 4 <= N_SEQUENCE <= 8:
            print(f'Invalid Number of Nodes in a Sequence: {N_SEQUENCE}. Please consider a number of Nodes between 4 and 8.')
        if 2 <= N_SEQUENCE <= 5:
            print(f'Invalid Error Target chosen: {N_SEQUENCE}. Please consider a number between 2 and 5.')
        if ((N_RECORDS//4) * N_SEQUENCE) > N_NODES:
            print(f'Invalid Error Inputs chosen: {N_RECORDS}, {N_SEQUENCE}. Please consider the inputs so they are ((N_RECORDS//4) * N_SEQUENCE) <= N_NODES.')
        raise

def main():
    TSP_FOLDER = "./TSPRandomGraph"
    DATASET_FOLDER = './Dataset'

    # Default Configuration if no args are given
    N_GRAPHS = 500
    N_NODES = 100
    N_RECORDS = 80
    N_SEQUENCE = 5
    ERROR_TARGET = 4
    parser = argparse.ArgumentParser(description='Dataset Configuration')
    parser.add_argument('--g', dest='graphs', type=int, help='Number of graphs to use')
    parser.add_argument('--r', dest='record', type=int, help='Record to Save for each Graph')
    parser.add_argument('--s', dest='sequence_nodes', type=int, help='Nodes in each sequence')
    parser.add_argument('--e', dest='error_target', type=float, help='Error target on the tour')
    args = parser.parse_args()
    # Set the Configuration Parameter
    if args.graphs is not None:
        N_GRAPHS = args.graphs
    if args.record is not None:
        N_RECORDS = args.record
    if args.sequence_nodes is not None:
        N_SEQUENCE = args.sequence_nodes
    if args.error_target is not None:
        ERROR_TARGET = args.error_target
    # Check if the Parameters are correct
    validate_input(N_NODES, N_SEQUENCE, N_RECORDS, ERROR_TARGET)

    print(f'Number of Records for each Graph: {N_RECORDS}, Number of Nodes in a Sequence: {N_SEQUENCE}')
    create_dataset(N_GRAPHS, TSP_FOLDER, DATASET_FOLDER, N_RECORDS, N_SEQUENCE, ERROR_TARGET)

if __name__ == "__main__":
    main()    