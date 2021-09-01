import numpy as np
from random import randint
from scipy.spatial import distance_matrix

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

""" Create the Dataset
"""
def create_dataset(tour, tour_length, distance_matrix, N_RECORDS):
    sorted_errors = get_sorted_errors(tour, tour_length, distance_matrix)
    worst_indexes = [ x for x in list(sorted_errors)[:5] ]
    best_indexes = [ x for x in list(sorted_errors)[-5:] ] 
    print(f'Best Indexes: {best_indexes} \nWorst Indexes: {worst_indexes}')
    for record_idx in range(0, N_RECORDS):
        if record_idx <= 4:
            random_node = randint(0,100)
            sequence = get_sequence(tour, random_node)
            """ TODO 
                (1)
                - Salvare sequenza sul Dataset (Panda)
                - Selezionare 5 Sequenze Ottime random (controllando di non prendere nodi appartenenti alle stesse sequenze)
                - Salvare sequenze sul dataset con valore 1

                (2)
                - Utilizzare best_indexes per generare soluzioni buone
                - Prendere la sequenza dove c'è stato lo scambio e salvarla sul dataset con valore 1

                (3)
                - Utilizzare worst_indexes per generare soluzioni pessime
                - Prendere la sequenza dove c'è stato lo scambio e salvarla sul dataset con valore 0

                (4)
                - Utilizzare worst_indexes per generare soluzioni pessime
                - Scambiare gli archi della sequenza contentente un worst index. Confrontare le diverse soluzioni e 
                    salvare la peggiore sul dataset con valore 0
                - Iterare il procedimento per i prossimi worst_indexes
            """
        ...

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
    print(f'ORIGINAL TOUR: {tour}')
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
    Return the error between the given tour and the best known
        - best_tour_length : the length of the best tour known
        - tour_length : the length of the tour to compare
"""
def get_tour_error(best_tour_length, tour_length):
    return round(((tour_length - best_tour_length)/tour_length)*100, 2)

def main():
    FOLDER_NAME = "./TSPRandomGraph"
    SEQUENCE_DIMENSION = 5
    N_RECORDS = 20
    file_name = 'problem_95'
    coords, distance_matrix = get_coordinates_from_file(FOLDER_NAME, file_name)
    tour_length, tour = get_solution_from_file(FOLDER_NAME, file_name)
    print(f'Problem: {file_name} \nTour Length: {tour_length} \nTOUR: {tour}')
    #create_dataset(tour, tour_length, distance_matrix, N_RECORDS)


if __name__ == "__main__":
    main()    