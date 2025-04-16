import json
import argparse

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import solvers


@dataclass
class CmdArgs:
    output_path: Path


@dataclass
class TestCmdArgs(CmdArgs):
    test_case: str
    choices: List[str]
    timeout: float
    uav_nums: List[int]
    task_nums: List[int]
    random_test_times: int
    hp_values: List
    show: bool


@dataclass
class PlotCmdArgs(CmdArgs):
    file_path: str
    x: str
    labels: List[str]
    choices: List[str]
    save_dir: Path
    show: bool


import framework.test as test


def init_cmd_args():
    parser = argparse.ArgumentParser(description="Coalition Formation Game Simulation")

    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # parser_test
    parser_test = subparsers.add_parser("test", help="test the solver")

    parser_test.add_argument("--test_case", type=str, help="path to the test case file")

    parser_test.add_argument("--uav_nums", nargs="+", type=int, default=[40], help="uav_num list")
    parser_test.add_argument(
        "--task_nums", nargs="+", type=int, default=[20], help="number of tasks"
    )
    parser_test.add_argument("--hp_values", nargs="+", help="hyper params values")
    parser_test.add_argument("--choices", nargs="+", type=str, help="choices of algorithms")

    parser_test.add_argument("--timeout", type=int, default=10, help="timeout for each algorithm")
    parser_test.add_argument(
        "--random_test_times", type=int, default=5, help="number of random test times"
    )
    parser_test.add_argument(
        "-o", "--output", type=Path, default=None, help="path to the output file"
    )
    parser_test.add_argument("--show", action="store_true", help="show the plot")
    # parser_show
    parser_plot = subparsers.add_parser("plot", help="show the results")
    parser_plot.add_argument("-f", "--file_path", type=Path, help="path to the results file")
    parser_plot.add_argument("-x", "--xlabel", type=str, default="uav_num", help="x axis")
    parser_plot.add_argument(
        "--labels", nargs="+", type=str, default=["elapsed_time"], help="labels"
    )
    parser_plot.add_argument("--choices", nargs="*", type=str, help="choices of algorithms")
    parser_plot.add_argument("--save_dir", type=Path, default=None, help="path to the output file")
    parser_plot.add_argument("--show", action="store_true", help="show the plot")
    args = parser.parse_args()
    return args


all_labels = [
    "elapsed_time",
    "completion_rate",
    "resource_use_rate",
    "total_distance",
    "total_energy",
    "total_exploss",
]


def run_test_driver(cmd_args: TestCmdArgs):
    solver_types = solvers.driver.get_SolverTypes(cmd_args.choices)
    if cmd_args.test_case in ["uav_num", "task_num"]:
        if cmd_args.test_case == "uav_num":
            task_num = 20
            if len(cmd_args.task_nums) == 1:
                task_num = cmd_args.task_nums[0]

            results = test.TestNums.run_vary_uav_nums(
                cmd_args.uav_nums,
                solver_types,
                task_num=task_num,
                test_times=cmd_args.random_test_times,
            )
        elif cmd_args.test_case == "task_num":
            uav_num = 40
            if len(cmd_args.uav_nums) == 1:
                uav_num = cmd_args.uav_nums[0]
            results = test.TestNums.run_vary_task_nums(
                cmd_args.task_nums,
                solver_types,
                uav_num=uav_num,
                test_times=cmd_args.random_test_times,
            )

    elif cmd_args.test_case.startswith("hyper_params."):
        if cmd_args.hp_values is None:
            raise ValueError("hp_values must be provided")
        hp_values = [float(v) for v in cmd_args.hp_values]
        results = test.TestHyperParams.run_vary_hyper_params(
            cmd_args.test_case,
            hp_values,
            solver_types,
            task_num=10,
            uav_num=100,
            test_times=cmd_args.random_test_times,
        )
    else:
        results = test.run_on_test_case(solver_types, cmd_args.test_case, cmd_args.show)

    comments = [
        f"test_case: {cmd_args.test_case}",
        f"choices: {cmd_args.choices}",
        f"uav_nums: {cmd_args.uav_nums}",
        f"task_nums: {cmd_args.task_nums}",
        f"random_test_times: {cmd_args.random_test_times}",
        f"hp_values: {cmd_args.hp_values}",
        f"results num: {len(results)}",
    ]
    if cmd_args.test_case.endswith(".json"):
        test_case_name = Path(cmd_args.test_case).stem
    else:
        test_case_name = cmd_args.test_case
    save_path = Path(f"./.results/results_{test_case_name}_{'_'.join(cmd_args.choices)}.yaml")
    if cmd_args.output_path is not None:
        save_path = cmd_args.output_path

    test.save_results(results, save_path, comments)
    print(f"results saved to {save_path}")


def run_show_driver(cmd_args: PlotCmdArgs):
    results = test.read_results(cmd_args.file_path)
    # test.save_results(results, "./.results.csv")
    # exit()
    labels = cmd_args.labels
    if len(labels) == 1 and cmd_args.labels[0] == "all":
        labels = all_labels
    test.visualize_results(
        results,
        x=cmd_args.x,
        labels=labels,
        choices=cmd_args.choices,
        save_dir=cmd_args.save_dir,
        show=cmd_args.show,
    )


def main():
    # parse args
    args = init_cmd_args()
    if args.command == "test":
        cmd_args = TestCmdArgs(
            test_case=args.test_case,
            choices=args.choices,
            timeout=args.timeout,
            uav_nums=args.uav_nums,
            task_nums=args.task_nums,
            random_test_times=args.random_test_times,
            hp_values=args.hp_values,
            output_path=args.output,
            show=args.show,
        )
        print(cmd_args)
        run_test_driver(cmd_args)
    elif args.command == "plot":
        cmd_args = PlotCmdArgs(
            output_path=None,
            file_path=args.file_path,
            x=args.xlabel,
            labels=args.labels,
            choices=args.choices,
            save_dir=args.save_dir,
            show=args.show,
        )
        print(cmd_args)
        run_show_driver(cmd_args)
    else:
        raise ValueError("Invalid command")


import cProfile

# 示例使用
if __name__ == "__main__":
    main()
    # cProfile.run("main()", "restats.iros")
