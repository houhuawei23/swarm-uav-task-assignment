from framework import (
    UAVManager,
    Task,
    TaskManager,
    HyperParams,
    CoalitionManager,
)

from framework.utils import *

from solvers import (
    EnumerationSolver,
    IROS2024_CoalitionFormationGame,
    ChinaScience2024_CoalitionFormationGame,
)

from solvers.icra2024 import AutoUAV


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

    uavs = [
        AutoUAV(
            id=1,
            position=[0, 0, 0],
            resources=[0, 0, 0],
            value=1.0,
            max_speed=1.0,
            mass=1.0,
            fly_energy_per_time=1.0,
            hover_energy_per_time=1.0,
        ),
        AutoUAV(
            id=2,
            position=[0, 0, 0],
            resources=[0, 0, 0],
            value=1.0,
            max_speed=1.0,
            mass=1.0,
            fly_energy_per_time=1.0,
            hover_energy_per_time=1.0,
        ),
        AutoUAV(
            id=3,
            position=[0, 0, 0],
            resources=[0, 0, 0],
            value=1.0,
            max_speed=1.0,
            mass=1.0,
            fly_energy_per_time=1.0,
            hover_energy_per_time=1.0,
        ),
    ]
    tasks = [
        Task(
            id=1,
            position=[0, 0, 0],
            required_resources=[1, 1, 1],
            time_window=[0, 100],
            threat=0.5,
            execution_time=1.0,
        ),
        Task(
            id=2,
            position=[1, 1, 1],
            required_resources=[1, 1, 1],
            time_window=[0, 100],
            threat=0.5,
            execution_time=1.0,
        ),
    ]
    uav_manager = UAVManager(uavs)
    task_manager = TaskManager(tasks)

    hyper_params = HyperParams(
        resources_num=3,
        map_shape=calculate_map_shape(uav_manager, task_manager),
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
    uav1.coalition_manager.plot_map(
        neighbor_uav_manager, task_manager, hyper_params, ".coalition.png", show=False
    )
