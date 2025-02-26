from uav import UAV, UAVManager
from task import Task, TaskManager
from game import CoalitionFormationGame
from coalition import CoalitionSet
import json
import subprocess
from task_assign import EnumerationAlgorithm
from utils import save_uavs_and_tasks

from multiprocessing import Process, Queue
import time


def run_enumeration(
    uav_manager,
    task_manager,
    resources_num,
    map_shape,
    gamma=0.1,
    result_queue: Queue = None,  # for return result
):
    print("---")
    print("Enumeration")
    enumeration_algorithm = EnumerationAlgorithm(
        uav_manager,
        task_manager,
        resources_num=resources_num,
        map_shape=map_shape,
        gamma=gamma,
    )
    start_time = time.time()
    best_assignment, best_score = enumeration_algorithm.solve()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if result_queue is not None:
        result_queue.put(elapsed_time)
    print(f"Elapsed Time: {elapsed_time}")
    print(f"Best Assignment: {best_assignment}")
    print(f"Best Score: {best_score}")

    enu_coalition_set = CoalitionSet(
        uav_manager, task_manager, assignment=best_assignment
    )
    enu_coalition_set.plot_map("enumeration_result.png")


def run_coalition_game(
    uav_manager,
    task_manager,
    resources_num,
    map_shape,
    gamma=0.1,
    result_queue: Queue = None,
):
    print("---")
    print("Coalition Game")
    coalition_set = CoalitionSet(uav_manager, task_manager)
    game = CoalitionFormationGame(
        uav_manager,
        task_manager,
        coalition_set,
        resources_num=resources_num,
        map_shape=map_shape,
        gamma=gamma,
    )

    # coalition_set.plot_map()
    start_time = time.time()
    game.run(debug=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if result_queue is not None:
        result_queue.put(elapsed_time)
    print(f"Coalition Game Result: {coalition_set}")
    coalition_set.plot_map("coalition_game_result.png", plot_unassigned=True)


def multi_processes_run(
    uav_manager,
    task_manager,
    resources_num,
    map_shape,
    gamma=0.1,
    timeout=10,
):
    q1 = Queue()  # 创建队列用于传递返回值
    q2 = Queue()
    # 启动两个进程
    p1 = Process(
        target=run_enumeration,
        args=(uav_manager, task_manager, resources_num, map_shape, gamma, q1),
    )
    p2 = Process(
        target=run_coalition_game,
        args=(uav_manager, task_manager, resources_num, map_shape, gamma, q2),
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


def simple_run(
    uav_manager,
    task_manager,
    resources_num,
    map_shape,
    gamma=0.1,
    timeout=10,
):
    run_enumeration(
        uav_manager,
        task_manager,
        resources_num,
        map_shape,
        gamma=gamma,
    )
    run_coalition_game(
        uav_manager,
        task_manager,
        resources_num,
        map_shape,
        gamma=gamma,
    )


def test_coalition(test_case_path="../tests/case1.json"):
    resources_num = 2
    map_shape = (20, 20, 0)
    gamma = 0.1

    # 加载 无人机 任务
    with open(test_case_path, "r") as f:
        data = json.load(f)

    uav_manager = UAVManager.from_dict(data["uavs"])
    task_manager = TaskManager.from_dict(data["tasks"])

    uav_manager.format_print()
    task_manager.format_print()

    # 将字典转换为 JSON 并保存到文件
    # save_uavs_and_tasks(uav_manager, task_manager, test_data_path)
    timeout = 10
    # multi_processes_run(
    #     uav_manager,
    #     task_manager,
    #     resources_num,
    #     map_shape,
    #     gamma=gamma,
    #     timeout=timeout,
    # )
    simple_run(
        uav_manager,
        task_manager,
        resources_num,
        map_shape,
        gamma=gamma,
    )


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
