import pytest
import numpy as np
from framework.base import Point
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager


# 测试 UAV 类
def test_uav():
    # 创建一个 UAV 对象
    uav = UAV(
        id=1,
        resources=[1.0, 2.0, 3.0],
        position=Point([0.0, 0.0, 0.0]),
        value=10.0,
        max_speed=5.0,
        mass=2.0,
        fly_energy_per_time=1.5,
        hover_energy_per_time=2.0,
    )

    # 测试 UAV 的属性
    assert uav.id == 1
    assert np.array_equal(uav.resources, np.array([1.0, 2.0, 3.0]))
    assert uav.position.tolist() == [0.0, 0.0, 0.0]
    assert uav.value == 10.0
    assert uav.max_speed == 5.0
    assert uav.mass == 2.0
    assert uav.fly_energy_per_time == 1.5
    assert uav.hover_energy_per_time == 2.0

    # 测试 UAV 的方法
    target_pos = Point([3.0, 4.0, 0.0])
    assert (
        uav.cal_fly_energy(target_pos) == 2.0 * (5.0 / 5.0) * 1.5
    )  # mass * distance / speed * energy_per_time
    assert uav.cal_hover_energy(10.0) == 2.0 * 10.0 * 2.0  # mass * time * energy_per_time

    # 测试 UAV 的 to_dict 和 from_dict 方法
    uav_dict = uav.to_dict()
    uav_from_dict = UAV.from_dict(uav_dict)
    assert uav == uav_from_dict


# 测试 Task 类
def test_task():
    # 创建一个 Task 对象
    task = Task(
        id=1,
        position=Point([1.0, 1.0, 0.0]),
        required_resources=[1.0, 1.0, 1.0],
        time_window=[0, 10],
        threat=0.5,
        execution_time=2,
    )

    # 测试 Task 的属性
    assert task.id == 1
    assert np.array_equal(task.required_resources, np.array([1.0, 1.0, 1.0]))
    assert task.position.tolist() == [1.0, 1.0, 0.0]
    assert task.time_window == [0, 10]
    assert task.threat == 0.5
    assert task.execution_time == 2

    # 测试 Task 的方法
    # assert task.brief_info() == "T_1(req=[1.0, 1.0, 1.0], tw=[0, 10], thr=0.5)"

    # 测试 Task 的 to_dict 和 from_dict 方法
    task_dict = task.to_dict()
    task_from_dict = Task.from_dict(task_dict)
    assert task == task_from_dict


def get_uavs():
    uavs = [
        UAV(
            id=1,
            resources=[1.0, 2.0, 3.0],
            position=Point([0.0, 0.0, 0.0]),
            value=10.0,
            max_speed=5.0,
            mass=2.0,
            fly_energy_per_time=1.5,
            hover_energy_per_time=2.0,
        ),
        UAV(
            id=2,
            resources=[4.0, 5.0, 6.0],
            position=Point([1.0, 1.0, 0.0]),
            value=20.0,
            max_speed=6.0,
            mass=3.0,
            fly_energy_per_time=1.6,
            hover_energy_per_time=2.1,
        ),
    ]
    return uavs


def get_tasks():
    tasks = [
        Task(
            id=1,
            position=Point([1.0, 1.0, 0.0]),
            required_resources=[1.0, 1.0, 1.0],
            time_window=[0, 10],
            threat=0.5,
            execution_time=2,
        ),
        Task(
            id=2,
            position=Point([2.0, 2.0, 0.0]),
            required_resources=[2.0, 2.0, 2.0],
            time_window=[5, 15],
            threat=0.6,
            execution_time=3,
        ),
    ]
    return tasks


# 测试 UAVManager 类
def test_uav_manager():
    # 创建多个 UAV 对象
    uavs = get_uavs()

    # 创建 UAVManager 对象
    uav_manager = UAVManager(uavs)

    # 测试 UAVManager 的方法
    assert len(uav_manager) == 2
    assert uav_manager.get(1) == uavs[0]
    assert uav_manager.get(2) == uavs[1]
    assert uav_manager.random_one() in uavs

    # 测试 UAVManager 的 to_dict_list 方法
    uav_manager_dict = uav_manager.to_dict_list()
    assert len(uav_manager_dict) == 2

    # 测试 UAVManager 的 from_dict 方法
    uav_manager_from_dict = UAVManager.from_dict(uav_manager_dict)
    assert len(uav_manager_from_dict) == 2


# 测试 TaskManager 类
def test_task_manager():
    # 创建多个 Task 对象
    tasks = get_tasks()

    new_task2 = Task(
        id=2,
        position=Point([2.0, 2.0, 0.0]),
        required_resources=[2.0, 2.0, 2.0],
        time_window=[5, 15],
        threat=0.6,
        execution_time=3,
    )
    assert new_task2 == tasks[1]

    # 创建 TaskManager 对象
    task_manager = TaskManager(tasks, 3)

    # 测试 TaskManager 的方法
    assert len(task_manager) == 2
    assert task_manager.get(1) == tasks[0]
    assert task_manager.get(2) == tasks[1]

    # 测试 TaskManager 的 to_dict_list 方法
    task_manager_dict = task_manager.to_dict_list()
    assert len(task_manager_dict) == 2

    # 测试 TaskManager 的 from_dict 方法
    task_manager_from_dict = TaskManager.from_dict(task_manager_dict, 3)
    assert len(task_manager_from_dict) == 2

    # 测试 TaskManager 的 max_execution_time 和 min_execution_time 属性
    assert task_manager.max_execution_time == 3
    assert task_manager.min_execution_time == 2
