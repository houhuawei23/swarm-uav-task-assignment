import numpy as np
from solvers.icra2024 import get_connected_components

def test_get_connected_components():
    coords = [(0, 0), (5, 5), (15, 15), (20, 20)]
    comm_dis = 8
    distance_matrix = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            distance_matrix[i, j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
    print(distance_matrix)
    components = get_connected_components(distance_matrix, comm_dis)
    print(components)
    assert components == [[0, 1], [2, 3]]
