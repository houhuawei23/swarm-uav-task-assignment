from uav import UAV, UAVManager
from task import Task, TaskManager
from game import CoalitionFormationGame
from coalition import CoalitionSet
import json
import subprocess
from task_assign import EnumerationAlgorithm
from utils import save_uavs_and_tasks


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
    game.run(debug=False)
    print(f"Coalition Game Result: {coalition_set}")
    coalition_set.plot_map("coalition_game_result.png")

    print("---")
    print("Enumeration")
    enumeration_algorithm = EnumerationAlgorithm(
        uav_manager,
        task_manager,
        resources_num=resources_num,
        map_shape=map_shape,
        gamma=gamma,
    )
    best_assignment, best_score = enumeration_algorithm.solve()

    print(f"Best Assignment: {best_assignment}")
    print(f"Best Score: {best_score}")

    enu_coalition_set = CoalitionSet(
        uav_manager, task_manager, assignment=best_assignment
    )
    enu_coalition_set.plot_map("enumeration_result.png")


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
