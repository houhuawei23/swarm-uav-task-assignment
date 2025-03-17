from typing import List, Type, Dict
from dataclasses import dataclass, field
import time
import json

import matplotlib.pyplot as plt

from .base import HyperParams, GenParams
from .uav import UAV, UAVManager, generate_uav_list, generate_uav_dict_list
from .task import Task, TaskManager, generate_task_list, generate_task_dict_list
from .coalition_manager import CoalitionManager
from .mrta_solver import MRTASolver
from .utils import evaluate_assignment, EvalResult, calculate_map_shape


def test_solver(
    SolverType: Type[MRTASolver],
    task_manager: TaskManager,
    uav_manager: UAVManager,
    hyper_params: HyperParams,
):
    coalition_mana: CoalitionManager = CoalitionManager(
        uav_manager.get_ids(), task_manager.get_ids()
    )
    solver = SolverType(uav_manager, task_manager, coalition_mana, hyper_params)

    start_time = time.time()
    solver.run_allocate()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # print(f"{SolverType.type_name()} Result: {coalition_mana}")

    eval_reuslt = evaluate_assignment(
        uav_manager, task_manager, coalition_mana.task2coalition, hyper_params.resources_num
    )
    eval_reuslt.elapsed_time = elapsed_time
    # eval_reuslt.solver_name = SolverType.type_name()
    # eval_reuslt.task2coalition = coalition_mana.get_task2coalition().copy()
    # print(f"Eval Result: {eval_reuslt}")
    # coalition_manager.plot_map(
    #     uav_manager, task_manager, hyper_params, cmd_args.output_path, plot_unassigned=True
    # )

    return coalition_mana, eval_reuslt


def run_test(solver_types: List[Type[MRTASolver]], test_case_path: str):
    results: List[EvalResult] = []
    with open(test_case_path, "r") as f:
        data = json.load(f)

    for solver_type in solver_types:
        print(f"Running {solver_type.type_name()}...")
        uav_manager = UAVManager.from_dict(data["uavs"], solver_type.uav_type())
        task_manager = TaskManager.from_dict(data["tasks"])
        # uav_manager.format_print()
        # task_manager.format_print()
        hyper_params = HyperParams(
            resources_num=data["resources_num"],
            map_shape=calculate_map_shape(uav_manager, task_manager),
            alpha=1.0,
            beta=10.0,
            gamma=0.05,
            mu=-1.0,
            max_iter=10,
        )

        coalition_mana, eval_reuslt = test_solver(
            solver_type, task_manager, uav_manager, hyper_params
        )
        results.append(eval_reuslt)
        coalition_mana: CoalitionManager
        coalition_mana.plot_map(
            uav_manager, task_manager, hyper_params, output_path=None, show=True
        )
    return results


@dataclass
class SaveResult:
    solver_name: str
    test_case_name: str  # "random" means randomly generated test case
    uav_num: int
    task_num: int
    hyper_params: HyperParams
    eval_result: EvalResult
    # task2coalition: Dict[int | None, List[int]]

    def to_dict(self):
        return {
            "solver_name": self.solver_name,
            "test_case_name": self.test_case_name,
            "uav_num": self.uav_num,
            "task_num": self.task_num,
            "hyper_params": self.hyper_params.to_dict(),
            "eval_result": self.eval_result.to_dict(),
            # "task2coalition": self.task2coalition,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            solver_name=data["solver_name"],
            test_case_name=data["test_case_name"],
            uav_num=data["uav_num"],
            task_num=data["task_num"],
            hyper_params=HyperParams.from_dict(data["hyper_params"]),
            eval_result=EvalResult.from_dict(data["eval_result"]),
            # task2coalition=data["task2coalition"],
        )


def run_vary_uav_nums(uav_nums: List[int], solver_types: List[Type[MRTASolver]]):
    """
    alg-num_uav-result
    results: Dict[alg, Dict[int, EvalResult]]
    for alg in algs:
        for uav_num in uav_nums:
            uavs = generate_uavs(uav_num)
            test_solver(...)
    """
    results = {solver_type.type_name(): dict() for solver_type in solver_types}
    task_num = 10
    gen_params = GenParams()
    for uav_num in uav_nums:
        print(f"Running uav_num={uav_num}")
        uav_dict_list = generate_uav_dict_list(uav_num)
        task_dict_list = generate_task_dict_list(task_num)
        for solver_type in solver_types:
            # uavs = generate_uav_list(uav_num, UAVType=solver_type.uav_type())
            print(f"Running {solver_type.type_name()}...")

            uav_manager = UAVManager.from_dict(uav_dict_list, solver_type.uav_type())
            task_manager = TaskManager.from_dict(task_dict_list)
            # uav_manager.format_print()
            # task_manager.format_print()
            hyper_params = HyperParams(
                resources_num=gen_params.resources_num,
                map_shape=calculate_map_shape(uav_manager, task_manager),
                alpha=1.0,
                beta=10.0,
                gamma=0.05,
                mu=-1.0,
                max_iter=10,
            )

            coalition_mana, eval_reuslt = test_solver(
                solver_type, task_manager, uav_manager, hyper_params
            )
            save_result = SaveResult(
                solver_name=solver_type.type_name(),
                test_case_name="random",
                uav_num=uav_num,
                task_num=task_num,
                # task2coalition=coalition_mana.get_task2coalition().copy(),
                hyper_params=hyper_params,
                eval_result=eval_reuslt,
            )
            results[solver_type.type_name()][uav_num] = save_result
            coalition_mana: CoalitionManager
            # coalition_mana.plot_map(
            #     uav_manager, task_manager, hyper_params, output_path=None, show=True
            # )

    # print(results)
    for solver_type_name, results_dict in results.items():
        print(f"{solver_type_name}:")
        for uav_num, result in results_dict.items():
            print(f"uav_num: {uav_num}, result: {result}")
        print()

    return results


def collect_items(result_dict: Dict[int, SaveResult], label="elapsed_time"):
    """
    completion_rate: float = 0.0
    resource_use_rate: float = 0.0
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_exploss: float = 0.0
    elapsed_time: float = 0.0
    """
    uav_nums = sorted(result_dict.keys())
    collections = None
    if label == "elapsed_time":
        collections = [result_dict[uav_num].eval_result.elapsed_time for uav_num in uav_nums]
    elif label == "completion_rate":
        collections = [result_dict[uav_num].eval_result.completion_rate for uav_num in uav_nums]
    elif label == "resource_use_rate":
        collections = [result_dict[uav_num].eval_result.resource_use_rate for uav_num in uav_nums]
    elif label == "total_distance":
        collections = [result_dict[uav_num].eval_result.total_distance for uav_num in uav_nums]
    elif label == "total_energy":
        collections = [result_dict[uav_num].eval_result.total_energy for uav_num in uav_nums]
    elif label == "total_exploss":
        collections = [result_dict[uav_num].eval_result.total_exploss for uav_num in uav_nums]
    else:
        raise ValueError(f"Unknown label: {label}")
    return collections


def visualize_results_beta(results: Dict[str, Dict[int, SaveResult]], labels=["elapsed_time"]):
    for label in labels:
        plt.figure(figsize=(10, 6))

        for solver_name, result_dict in results.items():
            uav_nums = sorted(result_dict.keys())
            # elapsed_times = [result_dict[uav_num].elapsed_time for uav_num in uav_nums]
            collections = collect_items(result_dict, label=label)
            plt.plot(uav_nums, collections, marker="o", label=solver_name)

        plt.xlabel("Number of UAVs")
        plt.ylabel(f"{label}")
        plt.title(f"Comparison of {label} over uav nums")
        plt.legend()
        plt.grid(True)
        plt.show()


from .utils import format_json


def save_results(
    results: Dict[str, Dict[int, SaveResult]],
    file_path: str,
):
    save_result_dict_list = []
    for solver_name, result_dict in results.items():
        for uav_num, result in result_dict.items():
            save_result_dict_list.append(result.to_dict())

    with open(file_path, "w") as f:
        json.dump(save_result_dict_list, f, indent=4)
    format_json(file_path)


def read_results(file_path: str) -> Dict[str, Dict[int, SaveResult]]:
    with open(file_path, "r") as f:
        save_result_dict_list = json.load(f)
    results = {}
    for save_result_dict in save_result_dict_list:
        solver_name = save_result_dict["solver_name"]
        uav_num = save_result_dict["uav_num"]
        result = SaveResult.from_dict(save_result_dict)
        if solver_name not in results:
            results[solver_name] = {}
        results[solver_name][uav_num] = result
    return results


class TestFramework:
    """
    指定方法列表和超参数，自动进行测试，得到每种方法的结果和评价指标，进行分析与可视化
    """

    def __init__(self, solver_types: List[Type[MRTASolver]]):
        self.solver_types: List[Type[MRTASolver]] = solver_types
        # self.hyper_params = hyper_params
