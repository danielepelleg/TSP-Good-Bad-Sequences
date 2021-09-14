from CreateDataset import get_coordinates_from_file, create_record, get_tour_length, get_sequence_length
from test_algorithms import KNNAlgorithm
from random import randint
from itertools import permutations
import argparse

""" TODO
    (0) Ottimizzare passaggio di parametri alle funzioni che ritornano l'algoritmo
        - Effettuare il dataset split una sola volta e passare il dataset diviso agli algoritmi
    (1) Provare Simulazioni di altri problemi
    (2) Provare Simulazione da file
    (3) Testare con grafo di dimensione maggiore
    (4) Migliorare algoritmo greedy affinchè ricerchi sequenze migliori prima di arrendersi
    (5) Aggiungere la funzione di plot
    (6) Commentare e rinominare files.py
"""

def get_node_neighbors(node, distance_matrix, solution, sequence):
    """ Get the neighbors of the given node

        The neighbors are the nodes not in the solution, close to the given one, to add to the path.
        Return a dictionary {node: distance} containing these nodes sorted from the closest to the farthest.
        
        :node: the node to consider as the reference
        :distance_matrix: the distance matrix of the nodes
        :solution: the current solution of the tour
        :sequence: the next sequence to add to the solution
    """
    neighbors = distance_matrix[node-1] # [0.5, 0, 0.65, 1.3, 3.6, ...]
    indexes = [i for i  in range(1, len(distance_matrix)+1)]
    neighbors_dictionary = dict(zip(indexes, neighbors))
    for n in solution:
        del neighbors_dictionary[n]
    for s in sequence:
        if s not in solution:
            del neighbors_dictionary[s]
    sorted_neighbors = {k: v for k, v in sorted(neighbors_dictionary.items(), key=lambda item: item[1])}
    return sorted_neighbors

def get_best_sequence_permutation(sequence, distance_matrix, limit):
    """ Get the best permutation of the last sequence

        :sequence: the sequence to permutate
        :distance_matrix: the distance matrix of the nodes
        :limit: the limit to use to choose the nodes of the sequence to permutate
    """
    #print(f'Permutating the Sequence: {sequence}\tLIMIT:{limit}')
    permutation_tuple = list(permutations(node for node in sequence[1:-limit]))
    permutation_list = [list(i) for i in permutation_tuple]
    for permutation in permutation_list:
        permutation.insert(0, sequence[0])
        for i in range(-limit, 0):
            permutation.append(sequence[i])
    index = 0
    for idx, permutation in enumerate(permutation_list):
        if idx == 0:
            min = get_sequence_length(permutation_list[idx], distance_matrix)
            continue
        current_sequence_length = get_sequence_length(permutation, distance_matrix)
        if min > current_sequence_length:
            min = current_sequence_length
            index = idx
    #print(f'Best Permutation: {permutation_list[index]}')
    return permutation_list[index] 

def get_best_permutations_sorted(sequence, distance_matrix):
    """ Get the best permutations of a sequence 

        Return a list containing the permutations of a sequence sorted from the best
        to the worst. The first permutations is the one with the lower cost.
        
        :sequence: the sequence to permutate
        :distance_matrix: the distance matrix of the nodes
        :return: list containing the permutations sorted
    """
    permutation_tuple = list(permutations(node for node in sequence[1:]))
    permutation_list = [list(i) for i in permutation_tuple]
    permutation_dict = {}
    for permutation in permutation_list:
        permutation.insert(0, sequence[0])
        permutation_dict[tuple(permutation)] = get_sequence_length(permutation, distance_matrix)
    permutation_dict_sorted = {k: v for k, v in sorted(permutation_dict.items(), key=lambda item: item[1])}
    return [list(p) for p in list(permutation_dict_sorted.keys())]

def add_sequence_to_solution(sequence, solution, limit=None):
    """ Add a Sequence to the Solution
    """
    for node in sequence[1:limit]:
        solution.append(node)

def greedy(coords, distance_matrix, starting_node=1, n=5):
    """ Greedy Algorithm

        The choices of the algorithm are guided by the scores of the ML algorithms.
        
        :coords: the list of the coords of the nodes
        :distance_matrix: the distance matrix of the nodes
        :starting_node: the node from which the solution starts
        :n: the number of the node in the sequence
    """
    # Insert the first node in the solution
    solution = [starting_node]
    knn = KNNAlgorithm()
    
    # Search Good Sequences to add to the solution
    while (len(solution) != len(distance_matrix)):
        complete = False
        # Set the first node in the sequence
        sequence = [starting_node]
        random_neighbor = starting_node
        
        # Build the Sequence
        while(len(sequence) < n):
            neighbors_dict = get_node_neighbors(random_neighbor, distance_matrix, solution, sequence)
            # Number of Neighbors to take
            target_range = round(30*len(neighbors_dict)/100)
            # Index of Random Neighbour
            k = randint(0, target_range)
            random_neighbor = list(neighbors_dict.keys())[k]
            # Insert Neighbour in the Sequence
            sequence.append(random_neighbor)
            
            # Last Sequence
            if (len(sequence)-1 + len(solution)) == len(distance_matrix):
                limit = n-len(sequence)     # Nodes in the solution used to close the sequence
                # Close the Sequence
                for i in range(0, limit):
                    sequence.append(solution[i])
                # Insert the best sequence permutation in the solution
                best_permutation = get_best_sequence_permutation(sequence, distance_matrix, limit)
                add_sequence_to_solution(best_permutation, solution, -limit)
                print(f'Complete!\n{solution}\n')
                complete = True
                break
        # Solution Completed
        if complete: 
            break
        
        # Get the Permutation List of sequences sorted from the best to the worst
        permutation_list_sorted = get_best_permutations_sorted(sequence, distance_matrix)
        for permutation in permutation_list_sorted:
            # Create the record to pass to the ML algorithm
            sequence_record = [create_record(permutation, coords, distance_matrix, None)]
            # Good Sequence -> add it to the solution
            if knn.predict(sequence_record)[0] == 1:
                add_sequence_to_solution(permutation, solution)
                print(f'New Sequence Appended: {permutation}')
                complete = True
                break
        # All the permutations are bad. Add the best to the solution.
        if not complete:
            print(f'Sono a {len(solution)}%')
            first_permutation = permutation_list_sorted[0]
            add_sequence_to_solution(first_permutation, solution)
            print(f'New Sequence Appended: {permutation}')
        # First Node in the Sequence = Last Node in the Solution
        starting_node = solution[-1]
    
    tour_length = get_tour_length(solution, distance_matrix)
    print(f'SOL. LENGTH: {len(solution)}')
    print(f'TOUR LENGTH: {tour_length}')

def main():
    # Default Configuration if no args are given
    TSP_FOLDER = "./TSPRandomGraph-Test"
    PROBLEM_NUMBER = 1
    N_SEQUENCE = 5
    starting_node = 1
    # Run's Configuration Settings
    parser = argparse.ArgumentParser(description='Dataset Configuration')
    parser.add_argument('--p', dest='problem', type=int, help='Number of the problem to use')
    parser.add_argument('--s', dest='sequence', type=int, help='Number of the sequence to use to solve the problem')
    parser.add_argument('--i', dest='starting_node', type=int, help='Number of the starting node to use')
    parser.add_argument('--file', dest='file', type=str, help='Name of the file to use')
    args = parser.parse_args()
    # Set the Configuration Parameter
    if args.problem is not None:
        PROBLEM_NUMBER = args.problem
    file_name = f'problem_{PROBLEM_NUMBER}'
    if args.sequence is not None:
        N_SEQUENCE = args.sequence
    if args.starting_node is not None:
        starting_node = args.starting_node
    if args.file is not None:
        TSP_FOLDER = f''
        file_name = args.file
    coords, distance_matrix = get_coordinates_from_file(TSP_FOLDER, file_name)
    greedy(coords, distance_matrix, starting_node, N_SEQUENCE)

if __name__ == "__main__":
    main()