from typing import List, Type, Dict, Tuple
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
from .utils import format_with_prettier
from . import utils
from . import task
from . import uav
from . import sim


import pandas as pd
import seaborn as sns
from tqdm import tqdm


def test_solver(
    SolverType: Type[MRTASolver],
    task_manager: TaskManager,
    uav_manager: UAVManager,
    hyper_params: HyperParams,
) -> Tuple[CoalitionManager, EvalResult]:
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


from pathlib import Path


def run_on_test_case(
    solver_types: List[Type[MRTASolver]],
    test_case_path: str | Path,
    show: bool = False,
    save_dir: Path = None,
    sim_on_step: bool = False,
):
    test_case_path = Path(test_case_path)
    print(f"Running on test case: {test_case_path}")
    results: List[EvalResult] = []
    with open(test_case_path, "r") as f:
        data = json.load(f)
    hyper_params = HyperParams(
        resources_num=data["resources_num"],
        map_shape=utils.calculate_map_shape_on_dict_list(data["uavs"], data["tasks"]),
        # resource_contribution_weight=10.0,
        # path_cost_weight=1.0,
        # threat_loss_weight=1.0,
        # zero_resource_contribution_penalty=-1.0,
        # resource_waste_weight=1.0,
        # max_iter=10,
    )
    for solver_type in solver_types:
        print(f"Running {solver_type.type_name()}...")
        uav_manager = UAVManager.from_dict(data["uavs"], solver_type.uav_type())
        task_manager = TaskManager.from_dict(data["tasks"], hyper_params.resources_num)

        coalition_mana, eval_reuslt = test_solver(
            solver_type, task_manager, uav_manager, hyper_params
        )
        save_result = SaveResult(
            solver_type.type_name(),
            test_case_path.stem,
            len(uav_manager.get_ids()),
            len(task_manager.get_ids()),
            hyper_params,
            eval_reuslt,
            task2coalition=coalition_mana.get_task2coalition().copy(),
        )
        results.append(save_result)
        # coalition_mana: CoalitionManager
        if save_dir is not None:
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"{solver_type.type_name()}_{test_case_path.stem}.png"
        else:
            output_path = None

        if show or (output_path is not None):
            coalition_mana.plot_map(
                uav_manager, task_manager, hyper_params, output_path=output_path, show=show
            )
        if sim_on_step:
            env = sim.SimulationEnv(uav_manager, task_manager, coalition_mana, hyper_params)
            env.run(10, debug_level=0)

    return results


@dataclass
class SaveResult:
    solver_name: str
    test_case_name: str  # "random" means randomly generated test case
    uav_num: int
    task_num: int
    hyper_params: HyperParams
    eval_result: EvalResult
    task2coalition: Dict[int, List[int]] = None

    def to_dict(self):
        return {
            "solver_name": self.solver_name,
            "test_case_name": self.test_case_name,
            "uav_num": self.uav_num,
            "task_num": self.task_num,
            "hyper_params": self.hyper_params.to_dict(),
            "eval_result": self.eval_result.to_dict(),
            "task2coalition": self.task2coalition,
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
            task2coalition=data.get("task2coalition", None),
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

    @classmethod
    def from_flattened_dict(cls, data: Dict):
        return cls(
            solver_name=data["solver_name"],
            test_case_name=data["test_case_name"],
            uav_num=data["uav_num"],
            task_num=data["task_num"],
            hyper_params=HyperParams.from_flattened_dict(
                {
                    key.split(".")[1]: value
                    for key, value in data.items()
                    if key.startswith("hyper_params.")
                }
            ),
            eval_result=EvalResult.from_dict(
                {
                    key.split(".")[1]: value
                    for key, value in data.items()
                    if key.startswith("eval_result.")
                }
            ),
            # task2coalition=data.get("task2coalition", None),
        )


import yaml
from pathlib import Path


def save_results(results: List[SaveResult], file_path: str | Path, comments: List[str] = []):
    if type(file_path) is str:
        file_path = Path(file_path)
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    save_result_dict_list = [result.to_dict() for result in results]
    with open(file_path, "w") as f:
        if file_path.suffix == ".json":
            json.dump(save_result_dict_list, f, indent=4)
        elif file_path.suffix == ".yaml":
            # add header comments
            for comment in comments:
                f.write(f"# {comment}\n")
            yaml.dump(save_result_dict_list, f)
        elif file_path.suffix == ".csv":
            df = pd.DataFrame([result.to_flattened_dict() for result in results])
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    format_with_prettier(file_path)


def read_results(
    file_path: str | Path,
) -> List[SaveResult]:
    if type(file_path) is str:
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    flattened = False
    with open(file_path, "r") as f:
        if file_path.suffix == ".json":
            save_result_dict_list = json.load(f)
        elif file_path.suffix == ".yaml":
            save_result_dict_list = yaml.safe_load(f)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            save_result_dict_list = df.to_dict(orient="records")
            flattened = True
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    results = []
    for save_result_dict in save_result_dict_list:
        if flattened:
            result = SaveResult.from_flattened_dict(save_result_dict)
        else:
            result = SaveResult.from_dict(save_result_dict)
        results.append(result)
    return results


# def visualize_results(
#     result_list: List[SaveResult],
#     x: str,
#     labels=["elapsed_time"],
#     choices: List[str] = [],
#     save_dir: Path = None,
#     show: bool = True,
# ):
#     result_fdict_list = [d.to_flattened_dict() for d in result_list]
#     df = pd.DataFrame(result_fdict_list)
#     for label in labels:
#         plt.figure(figsize=(12, 8))
#         # choices
#         if choices:
#             df = df[df["solver_name"].isin(choices)]
#         sns.boxplot(x=x, y=f"eval_result.{label}", hue="solver_name", data=df, palette="Set3")
#         # sns.boxplot(x=x, y=f"eval_result.{label}", hue="solver_name", data=df, palette="Set3")
#         # sns.violinplot(x=x, y=f"eval_result.{label}", hue="solver_name", data=df, split=True)

#         plt.title(
#             f"Boxplot of {label} by {x} and Solvers",
#             fontdict={"fontsize": 26, "fontweight": "bold"},
#         )
#         plt.xlabel(f"{x}", fontdict={"fontsize": 28})
#         plt.ylabel(f"{label}", fontdict={"fontsize": 28})
#         plt.legend(title="Solver Name", fontsize=22, title_fontsize=18, loc="upper left")
#         # 设置坐标轴数字大小, 加粗
#         plt.tick_params(axis="both", labelsize=24)
#         # tight
#         plt.tight_layout()

#         plt.grid(True)
#         if save_dir is not None:
#             if not save_dir.exists():
#                 save_dir.mkdir(parents=True)
#             plt.savefig(save_dir / f"{label}_{x}.png", dpi=600)  # high dpi
#         if show:
#             plt.show()


def visualize_results(
    result_list: List[SaveResult],
    x: str,
    labels=["elapsed_time"],
    choices: List[str] = [],
    save_dir: Path = None,
    show: bool = True,
):
    result_fdict_list = [d.to_flattened_dict() for d in result_list]
    df = pd.DataFrame(result_fdict_list)

    for label in labels:
        # Create figure with higher resolution
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

        # Filter data if choices provided
        if choices:
            df_filtered = df[df["solver_name"].isin(choices)]
        else:
            df_filtered = df

        # Set color palette
        # colors = sns.color_palette("husl", n_colors=len(df_filtered["solver_name"].unique()))
        # colors = sns.color_palette("pastel")
        colors = sns.color_palette()
        # Create box plot
        sns.boxplot(
            x=x,
            y=f"eval_result.{label}",
            hue="solver_name",
            data=df_filtered,
            palette=colors,
            width=0.7,
            showfliers=False,
            ax=ax,
        )

        # Add line plot connecting medians
        solvers = df_filtered["solver_name"].unique()
        categories = sorted(df_filtered[x].unique())

        for idx, solver in enumerate(solvers):
            solver_medians = []
            for cat in categories:
                median = df_filtered[
                    (df_filtered[x] == cat) & (df_filtered["solver_name"] == solver)
                ][f"eval_result.{label}"].median()
                solver_medians.append(median)

            # Plot lines connecting medians
            x_positions = range(len(categories))
            plt.plot(
                x_positions,
                solver_medians,
                "-o",
                linewidth=2.5,
                markersize=10,
                color=colors[idx],
                alpha=0.8,
                zorder=5,
                # label=f"{solver} (trend)"
            )

        # Enhance the plot appearance
        plt.title(
            f"{label.replace('_', ' ').title()} Distribution",
            fontdict={"fontsize": 24, "fontweight": "bold"},
            pad=20,
        )
        plt.xlabel(
            x.replace("_", " ").title(),
            fontdict={"fontsize": 20, "fontweight": "bold"},
            labelpad=15,
        )
        plt.ylabel(
            label.replace("_", " ").title(),
            fontdict={"fontsize": 20, "fontweight": "bold"},
            labelpad=15,
        )

        # Enhance legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            title="Solver Types",
            title_fontsize=16,
            fontsize=14,
            #  bbox_to_anchor=(1.05, 1),
            # loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            #  edgecolor='black'
        )

        # Customize grid and style
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tick_params(axis="both", labelsize=16)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("#333333")

        # Adjust layout
        plt.tight_layout()

        # Save and show
        if save_dir is not None:
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            plt.savefig(
                save_dir / f"{label}_{x}.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
        if show:
            plt.show()
        plt.close()  # Close the figure to free memory


class TestFramework:
    """
    指定方法列表和超参数，自动进行测试，得到每种方法的结果和评价指标，进行分析与可视化
    """


def random_test(task_num, uav_num, gen_params: GenParams, solver_type: Type[MRTASolver]):
    task_dict_list = generate_task_dict_list(task_num, task.TaskGenParams(**gen_params.__dict__))
    uav_dict_list = generate_uav_dict_list(uav_num, uav.UAVGenParams(**gen_params.__dict__))
    hyper_params = HyperParams(
        resources_num=gen_params.resources_num,
        map_shape=utils.calculate_map_shape_on_dict_list(uav_dict_list, task_dict_list),
    )
    uav_manager = UAVManager.from_dict(uav_dict_list, solver_type.uav_type())
    task_manager = TaskManager.from_dict(task_dict_list, hyper_params.resources_num)

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


class TestHyperParams(TestFramework):
    @staticmethod
    def run_vary_hyper_params(
        hp_choice: str,
        values: List[float],
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

                    task_manager = TaskManager.from_dict(task_dict_list, gen_params.resources_num)
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
