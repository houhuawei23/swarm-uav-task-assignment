import json
import argparse


from framework.base import HyperParams
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.utils import calculate_map_shape
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
        map_shape=calculate_map_shape(uav_manager, task_manager),
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


def main():
    parser = argparse.ArgumentParser(description="Coalition Formation Game Simulation")
    # test_case
    parser.add_argument(
        "--test_case",
        type=str,
        default="../tests/case1.json",
        help="path to the test case file",
    )
    parser.add_argument(
        "--choice",
        type=str,
        default="csci",
        help="choice of algorithm: enum, iros, csci",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./.images/.result.png",
        help="path to the output file",
    )
    parser.add_argument("--show", action="store_true", help="whether to show the result")

    # parse args
    args = parser.parse_args()
    cmd_args = CmdArgs(test_case_path=args.test_case, output_path=args.output, choice=args.choice)
    print(
        f"Using test case: {cmd_args.test_case_path}, output path: {cmd_args.output_path}, choice: {cmd_args.choice}"
    )

    if cmd_args.test_case_path == "uav_nums":
        solver_types = get_SolverTypes([cmd_args.choice])
        # test_framework = TestFramework(solver_types)
        uav_nums = [10, 20, 40, 80]
        # results = test_framework.run_vary_uav_nums(uav_nums)
        # test_framework.visualize_results(results)
        # from framework.test import run_vary_uav_nums, run_test, save_results, read_results
        # from framework.test import visualize_results_beta
        from framework.test import TestUAVNums, TestUAVNumsBeta

        results = TestUAVNumsBeta.run_vary_uav_nums(uav_nums, solver_types)
        # results1 = TestUAVNumsBeta.read_results("./.results/results_uavnum_vary_iros.json")
        # results2 = TestUAVNumsBeta.read_results("./.results/results_uavnum_vary_csci.json")
        # results = results1 + results2
        # results = TestUAVNumsBeta.read_results("./.results/results_uavnum_vary_all_3.json")
        labels = [
            "elapsed_time",
            "completion_rate",
            "resource_use_rate",
            "total_distance",
            "total_energy",
            "total_exploss",
        ]
        TestUAVNumsBeta.visualize_results_gamma(results, labels)
        # TestUAVNumsBeta.save_results(results, f"./.results/results_uavnum_vary_{cmd_args.choice}_3.json")
    else:
        test_old(cmd_args)
    # results = run_test(solver_types, cmd_args.test_case_path)
    # for result in results:
    #     result.format_print()
    # test_framework.analyze()
    # if args.show:
    #     for result in test_framework.results:


# 示例使用
if __name__ == "__main__":
    main()
