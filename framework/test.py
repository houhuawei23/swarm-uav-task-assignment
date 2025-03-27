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
from .utils import evaluate_assignment, EvalResult, calculate_map_shape_on_mana
from .utils import format_json
from . import utils
from . import task
from . import uav

import pandas as pd
import seaborn as sns
from tqdm import tqdm


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
        hyper_params = HyperParams(
            resources_num=data["resources_num"],
            map_shape=calculate_map_shape_on_mana(uav_manager, task_manager),
            alpha=5.0,
            beta=5.0,
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

    def to_flattened_dict(self):
        flattened_dict = {
            "solver_name": self.solver_name,
            "test_case_name": self.test_case_name,
            "uav_num": self.uav_num,
            "task_num": self.task_num,
        }

        for key, value in self.hyper_params.to_flattened_dict().items():
            flattened_dict[f"hyper_params.{key}"] = value
        for key, value in self.eval_result.to_flattened_dict().items():
            flattened_dict[f"eval_result.{key}"] = value
        return flattened_dict


class TestFramework:
    """
    指定方法列表和超参数，自动进行测试，得到每种方法的结果和评价指标，进行分析与可视化
    """

    @staticmethod
    def save_results(
        results: List[SaveResult],
        file_path: str,
    ):
        save_result_dict_list = [result.to_dict() for result in results]
        print(len(save_result_dict_list))
        with open(file_path, "w") as f:
            json.dump(save_result_dict_list, f, indent=4)
        format_json(file_path)

    @staticmethod
    def read_results(file_path: str) -> List[SaveResult]:
        with open(file_path, "r") as f:
            save_result_dict_list = json.load(f)
        results = []
        for save_result_dict in save_result_dict_list:
            result = SaveResult.from_dict(save_result_dict)

            results.append(result)
        return results

    @staticmethod
    def visualize_results(result_list: List[SaveResult], x: str, labels=["elapsed_time"]):
        result_fdict_list = [d.to_flattened_dict() for d in result_list]
        df = pd.DataFrame(result_fdict_list)
        for label in labels:
            plt.figure(figsize=(15, 10))
            sns.boxplot(x=x, y=f"eval_result.{label}", hue="solver_name", data=df, palette="Set3")
            # sns.violinplot(x=x, y=f"eval_result.{label}", hue="solver_name", data=df, split=True)

            plt.title(f"Boxplot of {label} by {x} and Solvers")
            plt.xlabel(f"{x}")
            plt.ylabel(f"{label}")
            plt.legend(title="Solver Name")
            plt.grid(True)
            plt.show()


def random_test(task_num, uav_num, gen_params: GenParams, solver_type: Type[MRTASolver]):
    task_dict_list = generate_task_dict_list(task_num, task.TaskGenParams(**gen_params.__dict__))
    uav_dict_list = generate_uav_dict_list(uav_num, uav.UAVGenParams(**gen_params.__dict__))
    hyper_params = HyperParams(
        resources_num=gen_params.resources_num,
        map_shape=utils.calculate_map_shape_on_dict_list(uav_dict_list, task_dict_list),
    )
    uav_manager = UAVManager.from_dict(uav_dict_list, solver_type.uav_type())
    task_manager = TaskManager.from_dict(task_dict_list)

    coalition_mana, eval_result = test_solver(solver_type, task_manager, uav_manager, hyper_params)

    save_result = SaveResult(
        solver_name=solver_type.type_name(),
        test_case_name="random",
        uav_num=len(uav_dict_list),
        task_num=len(task_dict_list),
        hyper_params=hyper_params,
        eval_result=eval_result,
    )
    return save_result


class TestNums(TestFramework):
    @staticmethod
    def run_vary_uav_nums(
        uav_nums: List[int],
        solver_types: List[Type[MRTASolver]],
        task_num: int = 20,
        test_times: int = 10,
    ) -> List[SaveResult]:
        results = []
        gen_params = GenParams()

        for uav_num in uav_nums:
            print(f"Running uav_num={uav_num}")
            for solver_type in solver_types:
                print(f" Running {solver_type.type_name()}...")
                for test_time in tqdm(range(test_times)):
                    save_result = random_test(task_num, uav_num, gen_params, solver_type)
                    results.append(save_result)

        return results

    @staticmethod
    def run_vary_task_nums(
        task_nums: List[int],
        solver_types: List[Type[MRTASolver]],
        uav_num: int = 10,
        test_times: int = 10,
    ) -> List[SaveResult]:
        results = []
        gen_params = GenParams()
        for task_num in task_nums:
            print(f"Running task_num={task_num}")
            for solver_type in solver_types:
                print(f" Running {solver_type.type_name()}...")
                for test_time in tqdm(range(test_times)):
                    save_result = random_test(task_num, uav_num, gen_params, solver_type)
                    results.append(save_result)

        return results

    @staticmethod
    def visualize_results(result_list: List[SaveResult], x="uav_num", labels=["elapsed_time"]):
        result_fdict_list = [d.to_flattened_dict() for d in result_list]
        df = pd.DataFrame(result_fdict_list)
        # print(df.info())
        # exit()
        for label in labels:
            plt.figure(figsize=(15, 10))
            sns.boxplot(x=x, y=f"eval_result.{label}", hue="solver_name", data=df, palette="Set3")
            # sns.violinplot(x=x, y=f"eval_result.{label}", hue="solver_name", data=df, split=True)

            plt.title(f"Boxplot of {label} by {x} and Solvers")
            plt.xlabel(f"{x}")
            plt.ylabel(f"{label}")
            plt.legend(title="Solver Name")
            plt.grid(True)

            plt.show()


class TestHyperParams(TestFramework):
    @staticmethod
    def run_vary_hyper_params(
        hp_choice: str,
        values: List,
        solver_types: List[Type[MRTASolver]],
        task_num: int = 10,
        uav_num: int = 50,
        test_times: int = 10,
    ):
        # if "." in hp_choice:
        #     hp_choice = f"hyper_params{hp_choice}"
        if hp_choice.startswith("hyper_params."):
            hp_choice = hp_choice[len("hyper_params.") :]
        if hp_choice not in HyperParams.__dict__.keys():
            raise ValueError(f"Invalid hyper parameter choice: {hp_choice}")

        results = []
        gen_params = GenParams()
        # task_dict_list = generate_task_dict_list(
        #     task_num, task.TaskGenParams(**gen_params.__dict__)
        # )
        # uav_dict_list = generate_uav_dict_list(uav_num, uav.UAVGenParams(**gen_params.__dict__))

        for value in values:
            print(f"Running {hp_choice}={value}")
            for solver_type in solver_types:
                print(f" Running {solver_type.type_name()}...")
                for test_time in tqdm(range(test_times)):
                    task_dict_list = generate_task_dict_list(
                        task_num, task.TaskGenParams(**gen_params.__dict__)
                    )
                    uav_dict_list = generate_uav_dict_list(
                        uav_num, uav.UAVGenParams(**gen_params.__dict__)
                    )

                    task_manager = TaskManager.from_dict(task_dict_list)
                    uav_manager = UAVManager.from_dict(uav_dict_list, solver_type.uav_type())
                    hyper_params = HyperParams(
                        **{hp_choice: value},
                        resources_num=gen_params.resources_num,
                        map_shape=utils.calculate_map_shape_on_dict_list(
                            uav_dict_list, task_dict_list
                        ),
                    )
                    # print(hyper_params)
                    coalition_mana, eval_result = test_solver(
                        solver_type, task_manager, uav_manager, hyper_params
                    )
                    save_result = SaveResult(
                        solver_name=solver_type.type_name(),
                        test_case_name="random",
                        uav_num=len(uav_dict_list),
                        task_num=len(task_dict_list),
                        hyper_params=hyper_params,
                        eval_result=eval_result,
                    )
                    results.append(save_result)

        return results
