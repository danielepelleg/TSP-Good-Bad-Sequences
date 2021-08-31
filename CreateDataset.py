import numpy as np
from scipy.spatial import distance_matrix

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

def get_solution_from_file(FOLDER_NAME, file_name):
    tour = 0
    solution = []
    with open(f'{FOLDER_NAME}/{file_name}.tour') as tour_file:
        for idx, line in enumerate(tour_file.readlines()):
            try:
                if idx == 1:
                    for x in line.split(): 
                        if x.isdigit():
                            tour = int(x)
                step = line.strip()
                if step.isdigit():
                    solution.append(int(step))

            except ValueError:
                pass

    return tour, solution

def random_solution(solution, tour):
    ...


def main():
    FOLDER_NAME = "./TSPRandomGraph"
    coords, distance_matrix = get_coordinates_from_file(FOLDER_NAME, 'problem_1')
    print(coords)
    tour, solution = get_solution_from_file(FOLDER_NAME, 'problem_1')
    print(tour, solution)
    random_solution = list(range(100))
    total_distance = 0
    for idx, node in enumerate(random_solution):
        if node != random_solution[-1]:
            total_distance += distance_matrix[node][random_solution[idx+1]]
        else: total_distance += distance_matrix[node][random_solution[0]]
    print(random_solution)
    print(int(total_distance))


if __name__ == "__main__":
    main()    