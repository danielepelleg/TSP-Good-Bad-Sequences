import numpy as np
import os
from scipy.spatial import distance_matrix
import argparse
from time import strftime

""" Create a fully connected Graph

    Return a fully connected graph of n nodes and its distance matrix.
"""
def get_graph_mat(n=10, size=1):
    """ 
        Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    coords = size * np.random.uniform(size=(n,2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat

""" Save the Configuration used on a config.txt file
"""
def save_config(FOLDER_NAME, NR_NODES, STEP_SIZE, SEED):
    today_date = strftime('%d-%m-%Y-%H.%M')
    with open(f'{FOLDER_NAME}/config.txt', 'w') as log_file:
        log_file.write(f'NR_NODES: {NR_NODES}\n')
        log_file.write(f'STEP SIZE: {STEP_SIZE}\n')
        log_file.write(f'SEED: {SEED}\n')
        log_file.write(f'DATE: {today_date}\n')
        log_file.close()

""" Delete all the graphs in the previous simulation
"""
def dump_graphs(FOLDER_NAME, NR_GRAPHS):
    """ 
        Utility function to clear the given folder from different graph files .tsp, .par, .tour
        If the folder doesn't exist, it is created.
    """
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    else:
        if os.path.exists(f'{FOLDER_NAME}/config.txt'):
            os.remove(f'{FOLDER_NAME}/config.txt')
        for i in range(1, NR_GRAPHS+1, 1):
            file_name = f'problem_{i}'
            if os.path.exists(f'{FOLDER_NAME}/{file_name}.tsp'):
                os.remove(f'{FOLDER_NAME}/{file_name}.tsp')
            if os.path.exists(f'{FOLDER_NAME}/{file_name}.par'):
                os.remove(f'{FOLDER_NAME}/{file_name}.par')
            if os.path.exists(f'{FOLDER_NAME}/{file_name}.tour'):
                os.remove(f'{FOLDER_NAME}/{file_name}.tour')

""" Save the graph.tsp in a file
"""
def save_tsp_graph(FOLDER_NAME, file_name, index, coords, size):
    """ 
        Utility function to save the fully connected graph in a folder as a TSP file
    """
    # Save the TSP file
    with open(f'{FOLDER_NAME}/{file_name}.tsp', 'w') as tsp_file:
        tsp_file.write(f'NAME: {file_name}\n')
        tsp_file.write(f'TYPE: TSP\n')
        tsp_file.write(f'COMMENT: 100-nodes problem {index} size-{size}\n')
        tsp_file.write(f'DIMENSION: 100\n')
        tsp_file.write(f'EDGE_WEIGHT_TYPE : EUC_2D\n')
        tsp_file.write(f'NODE_COORD_SECTION\n')
        for idx, coord in enumerate(coords):
            tsp_file.writelines(f'{idx+1} {coord[0]} {coord[1]}')
            tsp_file.write("\n")
        tsp_file.write('EOF\n')
        tsp_file.close()
    
    # Save the PAR file
    with open(f'{FOLDER_NAME}/{file_name}.par', 'w') as par_file:
        par_file.write(f'PROBLEM_FILE = {file_name}.tsp\n')
        par_file.write(f'MOVE_TYPE = 5\n')
        par_file.write(f'PATCHING_C = 3\n')
        par_file.write(f'PATCHING_A = 2\n')
        par_file.write(f'RUNS = 10\n')
        par_file.write(f'OUTPUT_TOUR_FILE = {file_name}.tour')
        par_file.close()

"""
    Create n graphs
"""
def create_graphs(nodes, graphs, FOLDER_NAME, STEP_SIZE):
    dump_graphs(FOLDER_NAME, graphs)
    size = 10
    for number in range(1, graphs+1):
        size += STEP_SIZE
        file_name = f'problem_{number}'
        coords, dist_mat = get_graph_mat(nodes, size)
        save_tsp_graph(FOLDER_NAME, file_name, number, coords, size)

def main():
    # Default Configuration if no args are given
    TEST_FOLDER = "./TSPRandomGraph-Test"
    FOLDER = "./TSPRandomGraph"
    STEP_SIZE = 5
    NR_NODES = 100  # Number of nodes
    NR_GRAPHS = 200 # Number of graphs
    SEED = 1  # A seed for the random number generator #40921
    np.random.seed(SEED)
    parser = argparse.ArgumentParser(description='Graphs Configuration')
    parser.add_argument('--t', dest='type', type=str, help='Type of Graphs to create ( Dataset | Test )')
    parser.add_argument('--s', dest='size', type=int, help='Step Size to add on every graph')
    parser.add_argument('--g', dest='graphs', type=int, help='Number of graphs to create')
    args = parser.parse_args()
    # Set the Configuration Parameter
    if args.type == 'test':
        FOLDER = TEST_FOLDER
    if args.size is not None:
        STEP_SIZE = args.size
    if args.graphs is not None:
        NR_GRAPHS = args.graphs
    create_graphs(NR_NODES, NR_GRAPHS, FOLDER, STEP_SIZE)
    save_config(FOLDER, NR_NODES, STEP_SIZE, SEED)

if __name__ == "__main__":
        main()    