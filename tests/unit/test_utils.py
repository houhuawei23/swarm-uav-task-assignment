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


from framework.test import SaveResult, SaveResultBeta
from framework.utils import flatten_dict, unflatten_dict
import pandas as pd


def format_print_dict(base_dict: dict, indent=1):
    indent_str = " " * indent
    print(indent_str + "{")
    for k, v in base_dict.items():
        if isinstance(v, dict):
            print(indent_str + f" {k}: ")
            format_print_dict(v, indent + 1)
        else:
            print(indent_str + f" {k}: {v}")
    print(indent_str + "}")


import matplotlib.pyplot as plt


def test_EvalResult():
    er = EvalResult(1, 2, 3)
    # print(er)
    # print(er.__dict__)
    json.dumps(er.to_dict())
    er_dict = er.to_dict()
    er_dict = {
        "completion_rate": 1,
    }
    new_er = EvalResult.from_dict(er_dict)
    # print(new_er.to_dict())
    hyper_params = HyperParams()
    task2coalition = {1: [2], None: [2, 3]}
    sr = SaveResult("csci", "random", 2, 2, hyper_params, er)
    print(sr.to_flattened_dict())
    # print(sr)
    # print(sr.to_dict())
    dumped = json.dumps(sr.to_dict())
    # print(dumped)
    # print(json.loads(dumped))
    new_sr = SaveResult.from_dict(json.loads(dumped))
    # print(new_sr)
    # print(new_sr.to_dict())
    sr_dict = sr.to_dict()
    # for k, v in sr_dict.items():
    # print(k, v)
    flattened_sr = flatten_dict(sr_dict)
    key_prefixes = ["hyper_params", "eval_result"]
    unflattened_st = unflatten_dict(flattened_sr, key_prefixes)
    # print("sr_dict:")
    # format_print_dict(sr_dict)
    # print("flattened_sr:")
    # format_print_dict(flattened_sr)
    # print("unflattened_st:")
    # format_print_dict(unflattened_st)
    # print()
    df = pd.DataFrame([flattened_sr, flattened_sr])
    df = pd.DataFrame([sr.to_flattened_dict(), sr.to_flattened_dict()])
    print(df)
    print(df.info())
    df.plot(x="uav_num", y="eval_result.completion_rate", kind="bar")
    plt.show()
    # save
    # df.to_csv("test.csv", index=False)
    # # read
    # new_df = pd.read_csv("test.csv")
    # print(new_df)

    # df = pd.DataFrame([sr_dict])
    # print(df)
    # print(df["hyper_params"])
    # sb = SaveResultBeta("csci", "random", 2, 2, hyper_params, er)
    # print(hyper_params)
    # print(er)
    # print(sb)
    # d1 = {}
    # d2 = {1: 5}
    # d1.update(d2)
    # print(d1)


if __name__ == "__main__":
    test_EvalResult()
