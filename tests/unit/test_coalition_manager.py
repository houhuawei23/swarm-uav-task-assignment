from framework.uav import UAV, UAVManager, generate_uav_dict_list
from framework.task import Task, TaskManager, generate_task_dict_list
from framework.coalition_manager import CoalitionManager
from framework import HyperParams
from framework.utils import calculate_map_shape_on_mana
# from test_uav_task import get_tasks, get_uavs

from copy import deepcopy

def test_coalition_manager():
    uavs = generate_uav_dict_list(3)
    tasks = generate_task_dict_list(3)
    uav_manager = UAVManager.from_dict(uavs)
    task_manager = TaskManager.from_dict(tasks)
    uavs = uav_manager.get_all()
    tasks = task_manager.get_all()
    uav_manager.format_print()
    task_manager.format_print()
    hyper_params = HyperParams(
        resources_num=3,
        map_shape=calculate_map_shape_on_mana(uav_manager, task_manager),
        resource_contribution_weight=1.0,
        path_cost_weight=10.0,
        threat_loss_weight=0.05,
        zero_resource_contribution_penalty=-1.0,
        max_iter=25,
    )
    coalition_manager = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())
    # coalition_manager.plot_map(uav_manager, task_manager, hyper_params, ".coalition.png")
    # coalition_manager.format_print()
    # 测试初始状态
    assert coalition_manager.get_unassigned_uav_ids() == [1, 2, 3]
    assert coalition_manager.get_coalition(1) == []
    assert coalition_manager.get_coalition(2) == []

    # 测试分配 UAV 到任务
    uav1 = uav_manager.get(1)
    task1 = task_manager.get(1)
    coalition_manager.assign(uav1.id, task1.id)
    assert coalition_manager.get_coalition(task1.id) == [uav1.id]
    assert coalition_manager.get_uav2task()[uav1.id] == task1.id
    assert coalition_manager.get_unassigned_uav_ids() == [2, 3]
    # coalition_manager.format_print()

    # 测试更新分配
    new_assignment = {1: [1], 2: [3], 3: [], None: [2]}
    coalition_manager.update_from_assignment(new_assignment, uav_manager)
    coalition_manager.format_print()
    new_assignment[None].append(1)
    
    print(new_assignment)
    coalition_manager.format_print()



    assert coalition_manager.get_coalition(1) == [1]
    assert coalition_manager.get_coalition(2) == [3]
    # print(f"dd: {coalition_manager.get_unassigned_uav_ids()}")
    assert coalition_manager.get_unassigned_uav_ids() == [2]
    # coalition_manager.format_print()

    # 测试合并 CoalitionManager
    another_coalition_manager = CoalitionManager(
        [uav.id for uav in uavs], [task.id for task in tasks]
    )
    another_coalition_manager.assign(2, 1)
    # print(another_coalition_manager.get_task2coalition())
    coalition_manager.merge_coalition_manager(another_coalition_manager)
    assert coalition_manager.get_coalition(1) == [1, 2]
    assert coalition_manager.get_unassigned_uav_ids() == []

    # 测试未分配的 UAV
    coalition_manager.unassign(1)
    assert coalition_manager.get_unassigned_uav_ids() == [1]

    # 测试获取任务 ID
    assert coalition_manager.get_taskid(2) == 1
    assert coalition_manager.get_taskid(1) is None

    # dcopy = coalition_manager.deepcopy()
    dcopy = deepcopy(coalition_manager)
    
    dcopy.unassign(2)
    
    print(
        coalition_manager,
        id(coalition_manager),
        id(coalition_manager.task2coalition),
        id(coalition_manager.uav2task),
    )
    print(
        dcopy,
        id(dcopy),
        id(dcopy.task2coalition),
        id(dcopy.uav2task),
    )
    assert id(coalition_manager) != id(dcopy)
    assert id(coalition_manager.task2coalition) != id(dcopy.task2coalition)


if __name__ == "__main__":
    test_coalition_manager()
