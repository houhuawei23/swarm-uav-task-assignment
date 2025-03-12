import numpy as np
import json
from framework.base import HyperParams
from framework.utils import EvalResult
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


from framework.test import SaveResult


def test_EvalResult():
    er = EvalResult(1, 2, 3)
    print(er)
    print(er.__dict__)
    json.dumps(er.to_dict())
    er_dict = er.to_dict()
    er_dict = {
        "completion_rate": 1,
    }
    new_er = EvalResult.from_dict(er_dict)
    print(new_er.to_dict())
    hyper_params = HyperParams()
    sr = SaveResult("csci", "random", 2, 2, hyper_params, er)
    print(sr)
    print(sr.to_dict())
    dumped = json.dumps(sr.to_dict())
    print(dumped)
    print(json.loads(dumped))
    new_sr = SaveResult.from_dict(json.loads(dumped))
    print(new_sr)
    print(new_sr.to_dict())


if __name__ == "__main__":
    test_EvalResult()
