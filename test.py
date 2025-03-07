import time
import json
from multiprocessing import Process, Queue
import argparse
from dataclasses import dataclass, field


from framework import (
    UAVManager,
    Task,
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

from solvers.icra2024 import AutoUAV


def test_autouav():
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
    uav1: AutoUAV = uav_manager.get(1)
    connected_uav_ids = [2]
    uav1.init(connected_uav_ids, task_manager)
    uav1.coalition_manager.plot_map(uav_manager, task_manager)


if __name__ == "__main__":
    test_autouav()
