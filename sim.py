import time
import json
from multiprocessing import Process, Queue
import argparse
from dataclasses import dataclass, field


from framework import (
    UAVManager,
    TaskManager,
    HyperParams,
    CoalitionManager,
)

from framework.utils import *

from solvers import (
    EnumerationAlgorithm,
    IROS2024_CoalitionFormationGame,
    ChinaScience2024_CoalitionFormationGame,
)


@dataclass
class CmdArgs:
    test_case_path: str
    output_path: str
    choice: str
    timeout: float = field(default=10)


def run_solver(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    cmd_args: CmdArgs,
    result_queue: Queue = None,
):
    print("---")
    coalition_manager = CoalitionManager(uav_manager, task_manager, hyper_params=hyper_params)
    if cmd_args.choice == "iros":
        solver = IROS2024_CoalitionFormationGame(
            uav_manager, task_manager, coalition_manager, hyper_params=hyper_params
        )
    elif cmd_args.choice == "csci":
        solver = ChinaScience2024_CoalitionFormationGame(
            uav_manager, task_manager, coalition_manager, hyper_params=hyper_params
        )
    elif cmd_args.choice == "enum":
        solver = EnumerationAlgorithm(uav_manager, task_manager, coalition_manager, hyper_params)
    else:
        raise ValueError("Invalid choice")

    # coalition_set.plot_map()
    start_time = time.time()
    solver.run_allocate(debug=False)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if result_queue is not None:
        result_queue.put(elapsed_time)

    print(f"{cmd_args.choice} Result: {coalition_manager}")

    eval_reuslt = evaluate_assignment(
        uav_manager, task_manager, coalition_manager.task2coalition, hyper_params.resources_num
    )
    print(f"Eval Result: {eval_reuslt}")
    coalition_manager.plot_map(cmd_args.output_path, plot_unassigned=True)


def simple_run(uav_manager, task_manager, hyper_params, cmd_args: CmdArgs):
    # run_enumeration(uav_manager, task_manager, hyper_params)
    # run_coalition_game(uav_manager, task_manager, hyper_params)
    run_solver(uav_manager, task_manager, hyper_params, cmd_args)


def multi_processes_run(uav_manager, task_manager, hyper_params: HyperParams, cmd_args: CmdArgs):
    q1 = Queue()  # 创建队列用于传递返回值
    q2 = Queue()
    # 启动两个进程
    p1 = Process(
        target=run_solver,
        args=(uav_manager, task_manager, hyper_params, cmd_args, q1),
    )
    p2 = Process(
        target=run_solver,
        args=(uav_manager, task_manager, hyper_params, cmd_args, q2),
    )

    p1.start()
    p2.start()
    p1.join(timeout=cmd_args.timeout)  # in seconds
    p2.join(timeout=cmd_args.timeout)
    if p1.is_alive():
        print("Enumeration process did not finish in time. Terminating...")
        p1.terminate()
        p1.join()
    else:
        elapsed_time = q1.get()
        print(f"Enumeration Elapsed Time: {elapsed_time}")

    if p2.is_alive():
        print("Coalition Game process did not finish in time. Terminating...")
        p2.terminate()
        p2.join()
    else:
        elapsed_time = q2.get()
        print(f"Coalition Game Elapsed Time: {elapsed_time}")


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

    uav_manager = UAVManager.from_dict(data["uavs"])
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
    run_solver(uav_manager, task_manager, hyper_params, cmd_args)


# 示例使用
if __name__ == "__main__":
    main()
