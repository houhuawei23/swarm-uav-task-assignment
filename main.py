import json
import argparse


from framework.base import HyperParams
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.utils import calculate_map_shape
from framework.sim import SimulationEnv


from solvers.driver import CmdArgs, run_solver
from solvers.icra2024 import AutoUAV


def simple_run(uav_manager, task_manager, hyper_params, cmd_args: CmdArgs):
    # run_enumeration(uav_manager, task_manager, hyper_params)
    # run_coalition_game(uav_manager, task_manager, hyper_params)
    run_solver(uav_manager, task_manager, hyper_params, cmd_args)


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

    # parse args
    args = parser.parse_args()
    cmd_args = CmdArgs(test_case_path=args.test_case, output_path=args.output, choice=args.choice)
    print(
        f"Using test case: {cmd_args.test_case_path}, output path: {cmd_args.output_path}, choice: {cmd_args.choice}"
    )

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
        gamma=0.05,
        mu=-1.0,
        max_iter=25,
    )

    print(hyper_params)

    uav_manager.format_print()
    task_manager.format_print()

    # 将字典转换为 JSON 并保存到文件
    # save_uavs_and_tasks(uav_manager, task_manager, test_data_path)
    # timeout = 10
    # multi_processes_run(uav_manager, task_manager, hyper_params, timeout=timeout)
    # simple_run(uav_manager, task_manager, hyper_params)
    coalition_manager = run_solver(uav_manager, task_manager, hyper_params, cmd_args)
    sim_env = SimulationEnv(uav_manager, task_manager, coalition_manager, hyper_params)
    sim_env.run()
    # sim_env.visualize_simulation()


# 示例使用
if __name__ == "__main__":
    main()
