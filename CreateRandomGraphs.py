import numpy as np
import os
from scipy.spatial import distance_matrix

def get_graph_mat(n=10, size=1):
    """ 
        Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    coords = size * np.random.uniform(size=(n,2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat

def dump_graphs(FOLDER_NAME):
    """ 
        Utility function to clear the given folder from different graph files .tsp, .par, .tour
        If the folder doesn't exist, it is created.
    """
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    else:
        for i in range(1, 101, 1):
            file_name = f'problem_{i}'
            if os.path.exists(f'{FOLDER_NAME}/{file_name}.tsp'):
                os.remove(f'{FOLDER_NAME}/{file_name}.tsp')
            if os.path.exists(f'{FOLDER_NAME}/{file_name}.par'):
                os.remove(f'{FOLDER_NAME}/{file_name}.par')
            if os.path.exists(f'{FOLDER_NAME}/{file_name}.tour'):
                os.remove(f'{FOLDER_NAME}/{file_name}.tour')
        

def save_tsp_graph(FOLDER_NAME, file_name, index, coords):
    """ 
        Utility function to save the fully connected graph in a folder as a TSP file
    """

    # Save the TSP file
    with open(f'{FOLDER_NAME}/{file_name}.tsp', 'w') as tsp_file:
        tsp_file.write(f'NAME: {file_name}\n')
        tsp_file.write(f'TYPE: TSP\n')
        tsp_file.write(f'COMMENT: 100-nodes problem {index}\n')
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
def create_graphs(nodes, graphs, FOLDER_NAME):
    dump_graphs(FOLDER_NAME)
    size = 10
    for number in range(1, graphs+1, 1):
        size += 10
        file_name = f'problem_{number}'
        coords, dist_mat = get_graph_mat(nodes, size)
        print(type(coords))
        print(coords)
        save_tsp_graph(FOLDER_NAME, file_name, number, coords)

def main():
    FOLDER_NAME = "./TSPRandomGraph"
    NR_NODES = 100  # Number of nodes
    NR_GRAPHS = 100 # Number of graphs
    SEED = 1  # A seed for the random number generator
    np.random.seed(SEED)
    create_graphs(NR_NODES, NR_GRAPHS, FOLDER_NAME)

if __name__ == "__main__":
        main()    