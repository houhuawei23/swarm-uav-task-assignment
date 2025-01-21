from uav import UAV, UAVManager
from task import Task, TaskManager
from game import CoalitionFormationGame


# 示例使用
if __name__ == "__main__":
    resources_num = 2
    map_shape = (20, 20, 0)
    gamma = 0.1

    # 初始化无人机
    uav1 = UAV(1, [5, 3], [0, 0, 0], 10, 20)
    uav2 = UAV(2, [3, 4], [10, 10, 0], 15, 25)
    uav3 = UAV(3, [2, 5], [20, 20, 0], 20, 30)
    uavs = [uav1, uav2, uav3]
    uav_manager = UAVManager(uavs)
    # 初始化任务
    task1 = Task(1, [4, 2], [5, 5, 0], [0, 100], 0.5)
    task2 = Task(2, [3, 3], [15, 15, 0], [0, 100], 0.7)
    tasks = [task1, task2]
    task_manager = TaskManager(tasks)
    game = CoalitionFormationGame(
        uav_manager,
        task_manager,
        resources_num=resources_num,
        map_shape=map_shape,
        gamma=gamma,
    )

    val = game.cal_resource_contribution(uav1, task1)
    cost = game.cal_path_cost(uav1, task1, val)
    threat = game.cal_threat_cost(uav1, task1)
    benefit = game.cal_uav_task_benefit(uav1, task1)
    print(f"{uav1}")
    print(f"{task1}")
    print(f"Resource contribution: {val: .2f}")
    print(f"Path cost: {cost: .2f}")
    print(f"Threat cost: {threat: .2f}")
    print(f"Benefit: {benefit: .2f}")

    game.plot_map()
    final_coalitions = game.run(debug=False)
    game.plot_map()
