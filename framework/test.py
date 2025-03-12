from typing import List, Type
import time
import json

from .base import HyperParams
from .uav import UAV, UAVManager
from .task import Task, TaskManager
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
    solver.run_allocate(debug=False)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # print(f"{SolverType.type_name()} Result: {coalition_mana}")

    eval_reuslt = evaluate_assignment(
        uav_manager, task_manager, coalition_mana.task2coalition, hyper_params.resources_num
    )
    eval_reuslt.elapsed_time = elapsed_time
    eval_reuslt.solver_name = SolverType.type_name()
    eval_reuslt.task2coalition = coalition_mana.get_task2coalition().copy()
    # print(f"Eval Result: {eval_reuslt}")
    # coalition_manager.plot_map(
    #     uav_manager, task_manager, hyper_params, cmd_args.output_path, plot_unassigned=True
    # )

    return coalition_mana, eval_reuslt


class TestFramework:
    """
    指定方法列表和超参数，自动进行测试，得到每种方法的结果和评价指标，进行分析与可视化
    """

    def __init__(self, solver_types: List[MRTASolver]):
        self.solver_types = solver_types
        # self.hyper_params = hyper_params
        self.results: List[EvalResult] = []

    def run_test(self, test_case_path: str):
        with open(test_case_path, "r") as f:
            data = json.load(f)

        for solver_type in self.solver_types:
            uav_manager = UAVManager.from_dict(data["uavs"], solver_type.uav_type())
            task_manager = TaskManager.from_dict(data["tasks"])
            uav_manager.format_print()
            task_manager.format_print()
            hyper_params = HyperParams(
                resources_num=data["resources_num"],
                map_shape=calculate_map_shape(uav_manager, task_manager),
                alpha=1.0,
                beta=10.0,
                gamma=0.05,
                mu=-1.0,
                max_iter=25,
            )

            coalition_mana, eval_reuslt = test_solver(
                solver_type, task_manager, uav_manager, hyper_params
            )
            self.results.append(eval_reuslt)
            coalition_mana: CoalitionManager
            coalition_mana.plot_map(
                uav_manager, task_manager, hyper_params, output_path=None, show=True
            )

    def analyze(self):
        for result in self.results:
            result.format_print()
