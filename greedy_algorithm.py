from create_dataset import get_coordinates_from_file, create_record, get_tour_length, get_sequence_length
from test_algorithms import KNNAlgorithm, dataset_setup, RandomForestAlgoritm, DecisionTreeAlgorithm, NaiveBayesAlgorithm, AdaBoostAlgorithm, LogisticRegressionAlgorithm
from random import randint, seed
from itertools import permutations
from csv import writer
import matplotlib.pyplot as plt
import argparse
import time
import os

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
    permutation_tuple = list(permutations(node for node in sequence[1:limit]))
    permutation_list = [list(i) for i in permutation_tuple]
    for permutation in permutation_list:
        permutation.insert(0, sequence[0])
        if limit is not None:
            for i in range(limit, 0):
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

def plot_solution(coords, solution, tour_length, file_name):
    """ Plot the Solution
    """
    solution = [i-1 for i in solution]
    plt.scatter(coords[:,0], coords[:,1])
    n = len(coords)
    
    for idx in range(n-1):
        i, next_i = solution[idx], solution[idx+1]
        plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8)
    
    i, next_i = solution[-1], solution[0]
    plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8)
    plt.plot(coords[solution[0], 0], coords[solution[0], 1], 'x', markersize=10)
    plt.title(f'{file_name}.tsp\nLength: {tour_length}')
    plt.savefig(f"./TSPGraph-Test/Greedy Results/{file_name}.png")

def save_tsp_solution(file_name, solution, tour_length, algorythm, neighbors):
    """ 
        Utility function to save the solution graph in a folder as a TOUR file
    """
    timestamp = time.strftime('%a %b %d %H:%M:%S %Y')
    # Save the TSP file
    with open(f'./TSPGraph-Test/Greedy Results/{file_name}_{algorythm}.tour', 'w') as sol_file:
        sol_file.write(f'NAME: {file_name}\n')
        sol_file.write(f'COMMENT: Length = {tour_length}\n')
        sol_file.write(f'COMMENT: Found by Greedy - Neighbors {neighbors}% [{algorythm}] {timestamp}\n')
        sol_file.write(f'TYPE: TOUR\n')
        sol_file.write(f'DIMENSION: {len(solution)}\n')
        sol_file.write(f'TOUR_SECTION\n')
        for node in solution:
            sol_file.writelines(f'{node}')
            sol_file.write("\n")
        sol_file.write(f'-1\n')
        sol_file.write('EOF\n')
        sol_file.close()

def save_run(file_name, N_SEQUENCE, solution, sol_length, SEED):
    record = [sol_length, str(solution), N_SEQUENCE]
    if not os.path.exists('./TSPGraph-Test/Simulations/'):
        os.makedirs('./TSPGraph-Test/Simulations/')
    else:
        if not os.path.exists(f'./TSPGraph-Test/Simulations/{file_name}_SEED{SEED}.csv'):
            columns = ['Length Solution', 'Solution', 'Sequence Length']
            with open(f'./TSPGraph-Test/Simulations/{file_name}_SEED{SEED}.csv', 'w', newline='') as csv_file:
                # creating a csv writer object 
                csvwriter = writer(csv_file, delimiter=';')
                # writing the fields 
                csvwriter.writerow(columns)
                csvwriter.writerow(record)
        else:
            with open(f'./TSPGraph-Test/Simulations/{file_name}_SEED{SEED}.csv', 'a', newline='') as csv_file:
                # Pass this file object to csv.writer() and get a writer object
                writer_object = writer(csv_file, delimiter=';')
            
                # Pass the list as an argument into the writerow()
                writer_object.writerow(record)  

def greedy(coords, distance_matrix, classifier, starting_node=1, N_SEQUENCE=5, N_NEIGHBORS = 8, SEED = 1):
    """ Greedy Algorithm

        The choices of the algorithm are guided by the scores of the ML algorithms.
        
        :coords: the list of the coords of the nodes
        :distance_matrix: the distance matrix of the nodes
        :starting_node: the node from which the solution starts
        :N_SEQUENCE: the number of the node in the sequence#
    """
    if SEED is not None: 
        # Set the Seed
        print(f'SEED: {SEED}')
        seed(SEED)
    # Insert the first node in the solution
    solution = [starting_node]
    timeout = 0
    
    # Search Good Sequences to add to the solution
    while (len(solution) != len(distance_matrix)):
        # Increment timeout
        timeout += 1
        complete = False
        # Set the first node in the sequence
        sequence = [starting_node]
        random_neighbor = starting_node
        
        # Build the Sequence
        while(len(sequence) < N_SEQUENCE):
            neighbors_dict = get_node_neighbors(random_neighbor, distance_matrix, solution, sequence)
            # Number of Neighbors to take
            target_range = round(N_NEIGHBORS*len(neighbors_dict)/100)
            # Index of Random Neighbour
            k = randint(0, target_range)
            random_neighbor = list(neighbors_dict.keys())[k]
            # Insert Neighbour in the Sequence
            sequence.append(random_neighbor)
            
            # Last Sequence
            if (len(sequence)-1 + len(solution)) == len(distance_matrix):
                print(f'SEQ. {sequence}')
                limit = N_SEQUENCE-len(sequence)     # Nodes in the solution used to close the sequence
                print(limit)
                # Close the Sequence
                for i in range(0, limit):
                    sequence.append(solution[i])
                print(sequence)
                # Insert the best sequence permutation in the solution
                if limit == 0:
                    limit = None
                else: limit = -limit
                best_permutation = get_best_sequence_permutation(sequence, distance_matrix, limit)
                add_sequence_to_solution(best_permutation, solution, limit)
                print(f'Last Sequence Appended: {best_permutation}\n\nSOLUTION: {solution}\nComplete!\n')
                complete = True
                break
        # Solution Completed
        if complete: 
            break
        
        print('Permutating Sequences ...')
        # Get the Permutation List of sequences sorted from the best to the worst
        permutation_list_sorted = get_best_permutations_sorted(sequence, distance_matrix)
        print('Done!')
        for permutation in permutation_list_sorted:
            # Create the record to pass to the ML algorithm
            sequence_record = [create_record(permutation, coords, distance_matrix, None)]
            # Good Sequence -> add it to the solution
            if classifier.predict(sequence_record)[0] == 1:
                add_sequence_to_solution(permutation, solution)
                print(f'New Permutated Sequence Appended: {permutation}')
                print(f'LEN. {len(solution)}')
                # Reset timeout
                timeout = 0
                complete = True
                break
        if (len(solution) >= (70*len(distance_matrix)//100) and not complete) or timeout > round(N_NEIGHBORS*(3)):
            if timeout > round(N_NEIGHBORS*(3)):
                print('Timeout Exceeded')
            # All the permutations are bad. Add the best to the solution.
            print(f'Solution at {(len(solution)*100)//len(distance_matrix)}%')
            first_permutation = permutation_list_sorted[0]
            add_sequence_to_solution(first_permutation, solution)
            print(f'Best Permutation Appended: {permutation}')
            # Reset timeout
            timeout = 0
        # First Node in the Sequence = Last Node in the Solution
        starting_node = solution[-1]
    
    tour_length = get_tour_length(solution, distance_matrix)
    print(f'SOL. LENGTH: {len(solution)}')
    print(f'TOUR LENGTH: {tour_length}')
    print(solution)
    return solution, tour_length    

def main():
    # Default Configuration if no args are given
    TSP_FOLDER = "./TSPGraph-Test"
    PROBLEM_NUMBER = 1
    N_SEQUENCE = 7
    N_NEIGHBORS = 5
    SEED = None
    starting_node = 1
    # Run's Configuration Settings
    parser = argparse.ArgumentParser(description='Dataset Configuration')
    parser.add_argument('--p', dest='problem', type=int, help='Number of the problem to use')
    parser.add_argument('--s', dest='sequence', type=int, help='Number of the sequence to use to solve the problem')
    parser.add_argument('--i', dest='starting_node', type=int, help='Number of the starting node of the solution')
    parser.add_argument('--S', dest='seed', type=int, help='Seed used for the choice of a random number')
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
        TSP_FOLDER = "./TSPGraph-Test/Problems"
        file_name = args.file
    if args.seed is not None:
        SEED = args.seed
    coords, distance_matrix = get_coordinates_from_file(TSP_FOLDER, file_name)
    N_RECORDS = (100//N_SEQUENCE)*4
    X_train, X_test, y_train, y_test = dataset_setup(N_RECORDS, N_SEQUENCE, 4.0)
    classifier = NaiveBayesAlgorithm(X_train, X_test, y_train, y_test)
    solution, tour_length = greedy(coords, distance_matrix, classifier, starting_node, N_SEQUENCE, N_NEIGHBORS, SEED)
    save_tsp_solution(file_name, solution, tour_length, type(classifier).__name__, N_NEIGHBORS)
    plot_solution(coords, solution, tour_length, file_name)
    save_run(file_name, N_SEQUENCE, solution, tour_length, SEED)

if __name__ == "__main__":
    main()