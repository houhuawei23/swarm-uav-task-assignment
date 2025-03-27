from framework import (
    UAVManager,
    Task,
    TaskManager,
    HyperParams,
    CoalitionManager,
)

from framework.utils import *

from solvers.icra2024 import AutoUAV

from framework.uav import UAV, UAVManager, generate_uav_list
from framework.task import Task, TaskManager, generate_task_list
from framework.coalition_manager import CoalitionManager
from framework import HyperParams
from framework.utils import calculate_map_shape_on_mana


def test_autouav():
    auto_uav_dict = {
        "id": 1,
        "position": [0, 0, 0],
        "resources": [0, 0, 0],
        "value": 1.0,
        "max_speed": 1.0,
        "mass": 1.0,
        "fly_energy_per_time": 1.0,
        "hover_energy_per_time": 1.0,
    }
    uav = AutoUAV.from_dict(auto_uav_dict)
    print(uav)
    uavs = generate_uav_list(4, UAVType=AutoUAV)
    tasks = generate_task_list(4)
    uav_manager = UAVManager(uavs)
    task_manager = TaskManager(tasks)

    hyper_params = HyperParams(
        resources_num=3,
        map_shape=calculate_map_shape_on_mana(uav_manager, task_manager),
        alpha=1.0,
        beta=10.0,
        gamma=0.05,
        mu=-1.0,
        max_iter=25,
    )
    uav1: AutoUAV = uav_manager.get(1)
    uav2: AutoUAV = uav_manager.get(2)
    connected_uav_ids = [2]
    neighbor_uav_manager = UAVManager([uav1, uav2])
    uav1.init(neighbor_uav_manager, task_manager, hyper_params)
    print(uav1.coalition_manager.brief_info())
    # uav1.coalition_manager.plot_map(
    #     neighbor_uav_manager, task_manager, hyper_params, ".coalition.png"
    # )


if __name__ == "__main__":
    test_autouav()
