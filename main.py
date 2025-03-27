import json
import argparse


from framework.base import HyperParams
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.utils import calculate_map_shape_on_mana
from framework.sim import SimulationEnv
# from framework.test import TestFramework

from solvers.driver import CmdArgs, run_solver, get_SolverTypes
from solvers.icra2024 import AutoUAV


def simple_run(uav_manager, task_manager, hyper_params, cmd_args: CmdArgs):
    # run_enumeration(uav_manager, task_manager, hyper_params)
    # run_coalition_game(uav_manager, task_manager, hyper_params)
    run_solver(uav_manager, task_manager, hyper_params, cmd_args)


def test_old(cmd_args: CmdArgs):
    with open(cmd_args.test_case_path, "r") as f:
        data = json.load(f)

    UAVType = UAV
    if cmd_args.choice == "icra":
        UAVType = AutoUAV
    print(f"Using UAV Type: {UAVType}")
    uav_manager = UAVManager.from_dict(data["uavs"], UAVType)
    task_manager = TaskManager.from_dict(data["tasks"])

    hyper_params = HyperParams(
        resources_num=data["resources_num"],
        map_shape=calculate_map_shape_on_mana(uav_manager, task_manager),
        alpha=1.0,
        beta=10.0,
        gamma=0.5,
        mu=-1.0,
        max_iter=10,
    )

    print(hyper_params)

    uav_manager.format_print()
    task_manager.format_print()

    # 将字典转换为 JSON 并保存到文件
    # save_uavs_and_tasks(uav_manager, task_manager, test_data_path)
    # timeout = 10
    # multi_processes_run(uav_manager, task_manager, hyper_params, timeout=timeout)
    # simple_run(uav_manager, task_manager, hyper_params)
    # coalition_manager = run_solver(uav_manager, task_manager, hyper_params, cmd_args)
    solver_types = get_SolverTypes([cmd_args.choice])

    from framework.test import run_test

    results = run_test(solver_types, cmd_args.test_case_path)
    # sim_env = SimulationEnv(uav_manager, task_manager, coalition_manager, hyper_params)
    # sim_env.run()
    # sim_env.visualize_simulation()


from framework.test import TestNums, TestHyperParams


def main():
    parser = argparse.ArgumentParser(description="Coalition Formation Game Simulation")
    # test_case
    parser.add_argument(
        "--test_case", type=str, default="../tests/case1.json", help="path to the test case file"
    )
    parser.add_argument(
        "--choice", type=str, default="csci", help="choice of algorithm: enum, iros, csci"
    )
    parser.add_argument(
        "--output", type=str, default="./.images/.result.png", help="path to the output file"
    )
    parser.add_argument("--uav_nums", nargs="+", type=int, default=[10, 40], help="uav_num list")
    parser.add_argument(
        "--task_nums", nargs="+", type=int, default=[10, 40], help="number of tasks"
    )
    parser.add_argument("--hp_values", nargs="+", help="hyper params values")

    parser.add_argument("--choices", nargs="+", type=str, help="choices of algorithms")
    parser.add_argument("--timeout", type=int, default=10, help="timeout for each algorithm")
    parser.add_argument("--show", action="store_true", help="whether to show the result")

    parser.add_argument(
        "--random_test_times", type=int, default=5, help="number of random test times"
    )

    # parse args
    args = parser.parse_args()
    cmd_args: CmdArgs = CmdArgs(
        test_case_path=args.test_case,
        output_path=args.output,
        choice=args.choice,
        choices=args.choices,
        timeout=args.timeout,
        uav_nums=args.uav_nums,
        task_nums=args.task_nums,
        random_test_times=args.random_test_times,
    )
    print(cmd_args)

    if cmd_args.test_case_path in ["uav_num", "task_num"]:
        solver_types = get_SolverTypes(cmd_args.choices)

        if cmd_args.test_case_path == "uav_num":
            results = TestNums.run_vary_uav_nums(
                cmd_args.uav_nums, solver_types, test_times=cmd_args.random_test_times
            )
        elif cmd_args.test_case_path == "task_num":
            results = TestNums.run_vary_task_nums(
                cmd_args.task_nums, solver_types, test_times=cmd_args.random_test_times
            )

        # results1 = TestUAVNumsBeta.read_results("./.results/results_uavnum_vary_iros.json")
        # results2 = TestUAVNumsBeta.read_results("./.results/results_uavnum_vary_csci.json")
        # results = results1 + results2
        # results = TestUAVNumsBeta.read_results("./.results/results_uavnum_vary_all_mar22.json")
        # results = TestUAVNumsBeta.read_results("./.results/results_uavnum_vary_all_acution.json")
        labels = [
            "elapsed_time",
            "completion_rate",
            "resource_use_rate",
            "total_distance",
            "total_energy",
            "total_exploss",
        ]
        TestHyperParams.visualize_results(results, x=cmd_args.test_case_path, labels=labels)
        # TestUAVNumsBeta.save_results(results, f"./.results/results_uavnum_vary_{cmd_args.choice}_acution.json")
    elif cmd_args.test_case_path.startswith("hyper_params."):
        solver_types = get_SolverTypes(cmd_args.choices)
        if args.hp_values is None:
            raise ValueError("hp_values must be provided")
        hp_values = [float(v) for v in args.hp_values]
        results = TestHyperParams.run_vary_hyper_params(
            cmd_args.test_case_path,
            hp_values,
            solver_types,
            task_num=10,
            uav_num=100,
            test_times=cmd_args.random_test_times,
        )
        labels = [
            "elapsed_time",
            "completion_rate",
            "resource_use_rate",
            "total_distance",
            "total_energy",
            "total_exploss",
        ]
        TestNums.visualize_results(results, x=cmd_args.test_case_path, labels=labels)
    else:
        test_old(cmd_args)


import cProfile

# 示例使用
if __name__ == "__main__":
    main()
    # cProfile.run("main()", "restats.iros")
