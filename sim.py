import time
import json
from multiprocessing import Process, Queue


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


def run_enumeration(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    result_queue: Queue = None,  # for return result
):
    print("---")
    print("Enumeration")
    enumeration_algorithm = EnumerationAlgorithm(uav_manager, task_manager, hyper_params)
    start_time = time.time()
    best_assignment, best_score = enumeration_algorithm.solve()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if result_queue is not None:
        result_queue.put(elapsed_time)
    print(f"Elapsed Time: {elapsed_time}")
    print(f"Best Assignment: {best_assignment}")
    print(f"Best Score: {best_score}")

    enu_coalition_set = CoalitionManager(
        uav_manager, task_manager, assignment=best_assignment, hyper_params=hyper_params
    )
    enu_coalition_set.plot_map(".enumeration_result.png")


def run_iros_coalition_game(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    result_queue: Queue = None,
):
    print("---")
    print("Coalition Game")
    coalition_manager = CoalitionManager(uav_manager, task_manager, hyper_params=hyper_params)
    game = IROS2024_CoalitionFormationGame(
        uav_manager, task_manager, coalition_manager, hyper_params=hyper_params
    )

    # coalition_set.plot_map()
    start_time = time.time()
    game.run_allocate(debug=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if result_queue is not None:
        result_queue.put(elapsed_time)
    print(f"Coalition Game Result: {coalition_manager}")

    eval_reuslt = evaluate_assignment(
        uav_manager, task_manager, coalition_manager.task2coalition, hyper_params.resources_num
    )
    print(f"Eval Result: {eval_reuslt}")
    coalition_manager.plot_map(".coalition_game_result.png", plot_unassigned=True)


def run_coalition_game(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    result_queue: Queue = None,
):
    print("---")
    print("Coalition Game")
    coalition_manager = CoalitionManager(uav_manager, task_manager, hyper_params=hyper_params)
    game = ChinaScience2024_CoalitionFormationGame(
        uav_manager, task_manager, coalition_manager, hyper_params=hyper_params
    )

    # coalition_set.plot_map()
    start_time = time.time()
    game.run_allocate(debug=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if result_queue is not None:
        result_queue.put(elapsed_time)
    print(f"Coalition Game Result: {coalition_manager}")

    eval_reuslt = evaluate_assignment(
        uav_manager, task_manager, coalition_manager.task2coalition, hyper_params.resources_num
    )
    print(f"Eval Result: {eval_reuslt}")
    coalition_manager.plot_map(".coalition_game_result.png", plot_unassigned=True)


def multi_processes_run(uav_manager, task_manager, hyper_params, timeout=10):
    q1 = Queue()  # 创建队列用于传递返回值
    q2 = Queue()
    # 启动两个进程
    p1 = Process(
        target=run_enumeration,
        args=(uav_manager, task_manager, hyper_params, q1),
    )
    p2 = Process(
        target=run_coalition_game,
        args=(uav_manager, task_manager, hyper_params, q2),
    )

    p1.start()
    p2.start()
    p1.join(timeout=timeout)  # in seconds
    p2.join(timeout=timeout)
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


def simple_run(uav_manager, task_manager, hyper_params):
    # run_enumeration(uav_manager, task_manager, hyper_params)
    # run_coalition_game(uav_manager, task_manager, hyper_params)
    run_iros_coalition_game(uav_manager, task_manager, hyper_params)


def test_coalition(test_case_path="../tests/case1.json"):
    # 加载 无人机 任务
    with open(test_case_path, "r") as f:
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
    timeout = 10
    # multi_processes_run(
    #     uav_manager,
    #     task_manager,
    #     hyper_params,
    #     timeout=timeout,
    # )
    simple_run(uav_manager, task_manager, hyper_params)


import argparse


def main():
    parser = argparse.ArgumentParser(description="Coalition Formation Game Simulation")
    # test_case
    parser.add_argument(
        "--test_case",
        type=str,
        default="../tests/case1.json",
        help="path to the test case file",
    )
    # parse args
    args = parser.parse_args()
    test_case_path = args.test_case
    print(f"Using test case: {test_case_path}")
    test_coalition(test_case_path)


# 示例使用
if __name__ == "__main__":
    main()
