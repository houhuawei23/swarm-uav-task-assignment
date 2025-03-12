import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List

from .base import HyperParams
from .uav import UAV, UAVManager
from .task import Task, TaskManager
from .coalition_manager import CoalitionManager

from dataclasses import dataclass


@dataclass
class SimState:
    time_step: int
    uav_dict_list: list


class SimulationEnv:
    """
    Manages the simulation environment, entities, and execution flow.
    模拟uav从初始位置出发，运动到目标Task位置
    模拟 n 步，每一步遍历所有的 uav，运动一步，记录更新后的uav状态
    记录什么：记录“格局”，即每一步的全局状态信息,or SimState
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_mana: CoalitionManager,
        hyper_params: HyperParams,
    ):
        self.uav_manager = uav_manager
        self.task_manager = task_manager
        self.coalition_manager = coalition_mana
        self.hyper_params = hyper_params
        self.time_step = 0
        self.sim_history: List[SimState] = []

    def run(self, steps: int = 10, debug_level=0):
        print("--- Simulation Started ---")
        uav_dict_list = self.uav_manager.to_dict_list()
        sim_state = SimState(self.time_step, uav_dict_list)
        self.sim_history.append(sim_state)
        for _ in range(steps):
            self.simulate_step()
        print("--- Simulation Finished ---")
        # print(self.sim_history)
        # for sim_state in self.sim_history:
        #     print(sim_state.time_step, sim_state.uav_dict_list)
        self.visualize_simulation()
        # self.analyze_results()

    def simulate_step(self, debug=False):
        self.time_step += 1
        unassigned_uav_ids = self.coalition_manager.get_unassigned_uav_ids()
        for uav in self.uav_manager.get_all():
            if uav.id in unassigned_uav_ids:
                continue
            # else
            taskid = self.coalition_manager.get_taskid(uav.id)
            task = self.task_manager.get(taskid)
            # print(uav.debug_info())
            uav.move_to(task.position, 0.1)
            # print(uav.debug_info())

        # record state
        uav_dict_list = self.uav_manager.to_dict_list()
        sim_state = SimState(self.time_step, uav_dict_list)
        self.sim_history.append(sim_state)

    def visualize_simulation(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        shapex, shapey, _ = self.hyper_params.map_shape
        ax.set_xlim(0, shapex * 1.1)
        ax.set_ylim(0, shapey * 1.1)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("UAV Simulation")

        uav_scat = ax.scatter([], [], c="blue", label="UAVs")
        task_scat = ax.scatter([], [], c="red", marker="x", label="Tasks")
        ax.legend()

        # Extract UAV initial positions
        uav_positions = [np.array(uav["position"])[:2] for uav in self.sim_history[0].uav_dict_list]
        uav_positions = np.array(uav_positions)

        # Extract UAV trajectories over time
        uav_trajectories = [
            [np.array(state.uav_dict_list[i]["position"])[:2] for state in self.sim_history]
            for i in range(len(uav_positions))
        ]

        # Extract Task positions (assume they don't move)
        task_positions = [np.array(task.position.xyz)[:2] for task in self.task_manager.get_all()]
        task_positions = np.array(task_positions)

        def update(frame):
            current_positions = np.array(
                [uav_trajectories[i][frame] for i in range(len(uav_trajectories))]
            )
            uav_scat.set_offsets(current_positions)
            return (uav_scat,)

        ani = FuncAnimation(fig, update, frames=len(self.sim_history), interval=500, blit=False)

        # Plot static task positions
        task_scat.set_offsets(task_positions)

        plt.show()
