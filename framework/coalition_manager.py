from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import warnings
import logging
import random
import time

# from copy import deepcopy
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from .base import plot_entities_on_axes, HyperParams
from .uav import UAV, UAVManager
from .task import Task, TaskManager
from .utils import evaluate_assignment


class CoalitionManager:
    # task2coalition: Dict[int, List[int]] = field(default_factory=dict, init=False)
    # uav2task: Dict[int, int] = field(default_factory=dict, init=False)
    # 不允许存在 task/uav 为 None 的情况，避免出现 bug
    # 使用 task_id = 0 表示 unassigned
    free_uav_task_id = 0

    def __init__(self, uav_ids: List[int], task_ids: List[int]):
        if self.free_uav_task_id not in task_ids:
            raise ValueError("unassigned_task_id = 0 must be in task_ids")
        # task -> coalition is empty (only 0 -> all UAV ids)
        self.task2coalition = {task_id: [] for task_id in task_ids}
        self.task2coalition[self.free_uav_task_id] = copy.deepcopy(uav_ids)
        # uav -> task is None
        self.uavid2taskid = {uav_id: self.free_uav_task_id for uav_id in uav_ids}

    def __str__(self):
        return str(self.task2coalition)

    def deepcopy(self):
        # copy = CoalitionManager([], [])
        # copy.task2coalition = copy.deepcopy(self.task2coalition)
        # copy.uav2task = copy.deepcopy(self.uav2task)
        # return copy
        return copy.deepcopy(self)

    def update_from_assignment(self, assignment: Dict[int, List[int]], uav_manager: UAVManager):
        self.task2coalition = copy.deepcopy(assignment)
        # self.task2coalition = assignment
        self.uavid2taskid.clear()

        assigned_uav_ids = set()
        for task_id, uav_ids in assignment.items():
            for uav_id in uav_ids:
                self.uavid2taskid[uav_id] = task_id
                assigned_uav_ids.add(uav_id)

        # update None coalition?
        # self.task2coalition[self.free_uav_task_id] = []
        # for uav_id in uav_manager.get_ids():
        #     if uav_id not in assigned_uav_ids:
        #         self.task2coalition[self.free_uav_task_id].append(uav_id)

    def assign(self, uav_id: int, task_id: int):
        """Assigns a UAV to a task, updating the coalitions dictionary.
        if task is None, unassign the uav.
        """
        if task_id == self.free_uav_task_id:
            # print(f"Assigning u{uav_id} to None")
            self.unassign(uav_id)
            return

        # print(f"Assigning u{uav_id} to t{task_id}")
        if self.uavid2taskid[uav_id] != self.free_uav_task_id:
            # 如果 UAV 已经被分配给其他任务，则先取消分配
            self.unassign(uav_id)

            # raise Exception(
            #     f"UAV {uav_id} has already been assigned to task {self.uav2task[uav_id]}"
            # )
        self.task2coalition[task_id].append(uav_id)
        self.task2coalition[self.free_uav_task_id].remove(uav_id)
        self.uavid2taskid[uav_id] = task_id

    def unassign(self, uav_id: int):
        """Unassigns a UAV from its current task, updating the coalitions dictionary."""
        task_id = self.uavid2taskid[uav_id]
        # print(f"Unassigning u{uav_id} from t{task_id}")
        if task_id == self.free_uav_task_id:
            # 如果 UAV 未被分配给任何任务，则不进行任何操作
            # print(f"Warning: UAV {uav_id} is not assigned to any task")
            # warnings.warn(f"Warning: UAV {uav_id} is not assigned to any task", UserWarning)
            return
        else:
            self.task2coalition[task_id].remove(uav_id)
            self.task2coalition[self.free_uav_task_id].append(uav_id)
            self.uavid2taskid[uav_id] = self.free_uav_task_id

    def merge_coalition_manager(self, cmana: "CoalitionManager"):
        """
        1. cmana.uav2task[uavid] == self.uav2task[uav_id]: pass
        2. cmana.uav2task[uavid] != self.uav2task[uav_id]:
            2.1 cmana.uav2task[uavid] is None, self.uav2task[uav_id] is not None
            2.2 cmana.uav2task[uavid] is not None, self.uav2task[uav_id] is None
            2.3 cmana.uav2task[uavid] is not None, self.uav2task[uav_id] is not None

        """
        # print(f"self: {self.uav2task}")
        # print(f"self: {self.task2coalition}")
        # print(f"cmana: {cmana.uav2task}")
        # print(f"cmana: {cmana.task2coalition}")
        for uav_id, task_id in cmana.get_uav2task().items():
            # if task_id is None and self.uav2task[uav_id] is not None:
            if task_id == self.uavid2taskid[uav_id]:
                continue
            # task_id != self.uav2task[uav_id]
            if (task_id == self.free_uav_task_id) and (
                self.uavid2taskid[uav_id] != self.free_uav_task_id
            ):
                # self.assign(uav_id, task_id)
                continue
            elif (task_id != self.free_uav_task_id) and (
                self.uavid2taskid[uav_id] == self.free_uav_task_id
            ):
                self.assign(uav_id, task_id)
            else:  # (task_id is not None) and (self.uav2task[uav_id] is not None):
                raise Exception("Cannot merge coalition managers with conflicting assignments")

    def get_unassigned_uav_ids(self) -> List[int]:
        return self.task2coalition[self.free_uav_task_id]

    def get_coalition(self, task_id: int) -> List[int]:
        return self.task2coalition[task_id]

    def get_coalition_by_uav_id(self, uav_id: int) -> List[int]:
        return self.task2coalition[self.uavid2taskid[uav_id]]

    def get_taskid(self, uavid) -> int:
        """ """
        # assert uavid in self.uav2task.keys()
        if uavid not in self.uavid2taskid.keys():
            raise Exception(f"UAV {uavid} is not in the coalition manager")
        return self.uavid2taskid[uavid]

    def get_task2coalition(self) -> Dict[int, List[int]]:
        return self.task2coalition

    def get_uav2task(self) -> Dict[int, int]:
        return self.uavid2taskid

    def format_print(self):
        print(f"task2coalition: {self.task2coalition}")
        print(f"uav2task: {self.uavid2taskid}")

    def brief_info(self):
        info = f"task2coalition: {self.task2coalition}, "
        info += f"uav2task: {self.uavid2taskid}"
        return info

    def plot_coalition(
        self,
        ax: plt.Axes,
        task_id: int,
        coalition: List[int],
        uav_manager: UAVManager,
        task_manager: TaskManager,
        color_idx: int = 0,
    ):
        task = task_manager.get(task_id)

        # Use a more professional color palette
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, 20))
        coalition_color = colors[color_idx % 20]
        
        # Draw a more elegant enclosure around UAVs in the same coalition
        if len(coalition) > 1:
            x_coords = [uav_manager.get(uav_id).position.x for uav_id in coalition]
            y_coords = [uav_manager.get(uav_id).position.y for uav_id in coalition]
            x_coords.append(task.position.x)
            y_coords.append(task.position.y)
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            radius = max(np.max(x_coords) - center_x, np.max(y_coords) - center_y) + 3

            # Create a more elegant coalition boundary with gradient alpha
            circle = Circle(
                (center_x, center_y),
                radius,
                color=coalition_color,
                fill=True,
                alpha=0.1,
                linewidth=1.5,
                linestyle='-',
                zorder=1
            )
            ax.add_patch(circle)
            
            # Add a subtle outer ring for depth
            outer_circle = Circle(
                (center_x, center_y),
                radius + 0.5,
                color=coalition_color,
                fill=False,
                alpha=0.3,
                linewidth=0.8,
                linestyle='--',
                zorder=1
            )
            ax.add_patch(outer_circle)

        # Plot UAVs with enhanced styling
        uav_list = [uav_manager.get(uav_id) for uav_id in coalition]
        for uav in uav_list:
            x, y, z = uav.position.xyz
            
            # Add a glow effect for UAVs
            glow = plt.Circle((x, y), 0.7, color=coalition_color, alpha=0.2, zorder=2)
            ax.add_patch(glow)
            
            # Plot the UAV with a more professional style
            ax.scatter(x, y, color=coalition_color, marker='o', s=80, 
                      edgecolor='white', linewidth=1.5, zorder=3)
            
            # Add UAV label with improved styling
            text_delta = 0.3
            ax.text(x, y + text_delta, s=uav.brief_info(), 
                   ha="center", color=coalition_color, fontsize=10,
                   fontweight='medium', bbox=dict(facecolor='white', alpha=0.7, 
                                                 edgecolor=coalition_color, boxstyle='round,pad=0.2'),
                   zorder=4)

        # Draw improved arrows from UAVs to task with curved paths
        for uav_id in coalition:
            uav = uav_manager.get(uav_id)
            start_point = np.array([uav.position.x, uav.position.y])
            end_point = np.array([task.position.x, task.position.y])

            # Calculate vector from start to end
            delta = end_point - start_point
            dist = np.linalg.norm(delta)
            
            if dist < 0.1:  # Skip if UAV is too close to task
                continue
                
            unit_delta = delta / dist

            # Create perpendicular vector for control point
            perp = np.array([-unit_delta[1], unit_delta[0]])

            # Create a more natural curve with variable height based on distance
            curve_height = min(0.3 * dist, 5.0)  # Cap maximum curve height
            control_point = (start_point + end_point) / 2 + perp * curve_height

            # Create a smoother Bezier curve with more points
            t = np.linspace(0, 1, 50)
            bezier_points = np.outer((1-t)**2, start_point) + \
                           np.outer(2*(1-t)*t, control_point) + \
                           np.outer(t**2, end_point)
            
            # Draw a gradient arrow with fading effect
            for i in range(len(bezier_points) - 1):
                alpha = 0.4 + 0.4 * (i / len(bezier_points))  # Gradually increase opacity
                width = 1.0 + 0.5 * (i / len(bezier_points))  # Gradually increase width
                
                line = plt.Line2D([bezier_points[i][0], bezier_points[i+1][0]],
                                 [bezier_points[i][1], bezier_points[i+1][1]],
                                 color=coalition_color, alpha=alpha, linewidth=width,
                                 solid_capstyle='round', zorder=2)
                ax.add_line(line)

            # Add an elegant arrow head
            arrow_direction = end_point - bezier_points[-2]
            arrow_direction = arrow_direction / np.linalg.norm(arrow_direction)

            arrow_size = 0.8
            arrow_angle = np.arctan2(arrow_direction[1], arrow_direction[0])
            
            # Create a more professional arrow head
            arrow_head = Polygon(
                [
                    end_point,
                    end_point - arrow_size * np.array([np.cos(arrow_angle + np.pi/6), 
                                                      np.sin(arrow_angle + np.pi/6)]),
                    end_point - 0.5 * arrow_size * np.array([np.cos(arrow_angle), 
                                                            np.sin(arrow_angle)]),
                    end_point - arrow_size * np.array([np.cos(arrow_angle - np.pi/6), 
                                                      np.sin(arrow_angle - np.pi/6)]),
                ],
                color=coalition_color,
                alpha=0.8,
                zorder=3
            )
            ax.add_patch(arrow_head)

    def plot_map(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        hyper_params: HyperParams = None,
        output_path=None,
        plot_unassigned=True,
        show=True,
    ):
        # Create a more professional figure with higher resolution
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 12), dpi=120)
        
        # Set background and style
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # Add subtle grid for better readability
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Plot tasks with improved styling
        task_list = [task for task in task_manager.get_all() 
                    if task.id != task_manager.free_uav_task_id]
        
        # Plot tasks with enhanced markers
        for task in task_list:
            x, y, z = task.position.xyz
            # Add task highlight/glow
            task_glow = plt.Circle((x, y), 1.2, color='red', alpha=0.15, zorder=1)
            ax.add_patch(task_glow)
            
            # Plot task with professional styling
            ax.scatter(x, y, color='red', marker='s', s=120, 
                      edgecolor='white', linewidth=1.5, zorder=2)
            
            # Add task label with improved styling
            text_delta = 0.4
            ax.text(x, y + text_delta, s=task.brief_info(), 
                   ha="center", color='darkred', fontsize=11,
                   fontweight='medium', bbox=dict(facecolor='white', alpha=0.8, 
                                                 edgecolor='red', boxstyle='round,pad=0.3'),
                   zorder=3)

        # Plot UAVs and their coalitions with different colors
        coalition_idx = 0
        for task_id, coalition in self.task2coalition.items():
            if task_id == self.free_uav_task_id:
                continue
            self.plot_coalition(ax, task_id, coalition, uav_manager, task_manager, coalition_idx)
            coalition_idx += 1

        # Plot unassigned UAVs with improved styling
        if plot_unassigned:
            unassigned_uav_ids = self.get_unassigned_uav_ids()
            unassigned_uavs = [uav_manager.get(uav_id) for uav_id in unassigned_uav_ids]
            
            for uav in unassigned_uavs:
                x, y, z = uav.position.xyz
                
                # Add subtle glow for unassigned UAVs
                glow = plt.Circle((x, y), 0.7, color='gray', alpha=0.15, zorder=1)
                ax.add_patch(glow)
                
                # Plot the UAV with a more professional style
                ax.scatter(x, y, color='gray', marker='o', s=80, 
                          edgecolor='white', linewidth=1.5, alpha=0.7, zorder=2)
                
                # Add UAV label with improved styling
                text_delta = 0.3
                ax.text(x, y + text_delta, s=uav.brief_info(), 
                       ha="center", color='dimgray', fontsize=10,
                       fontweight='medium', bbox=dict(facecolor='white', alpha=0.7, 
                                                     edgecolor='lightgray', boxstyle='round,pad=0.2'),
                       zorder=3)
                
        # Add evaluation results with improved styling
        if hyper_params:
            resources_num = hyper_params.resources_num
        else:
            resources_num = len(uav_manager.random_one().resources)

        eval_result = evaluate_assignment(
            uav_manager,
            task_manager,
            self.task2coalition,
            resources_num,
        )

        # Create a professional stats box
        stats_text = (
            f"Task Completion Rate: {eval_result.completion_rate:.2f}\n"
            f"Resource Utilization: {eval_result.resource_use_rate:.2f}\n"
            f"Total UAVs: {len(uav_manager)}\n"
            f"Total Tasks: {len(task_manager)-1}"  # Subtract free UAV task
        )
        
        # Add a professional stats panel
        stats_box = plt.text(
            0.97, 0.03, stats_text,
            transform=ax.transAxes,
            fontsize=12,
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor='white',
                edgecolor='lightgray',
                alpha=0.8
            ),
            zorder=5
        )

        # Add title and labels with professional styling
        ax.set_xlabel("X Position", fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel("Y Position", fontsize=14, fontweight='bold', labelpad=10)
        
        # Calculate appropriate axis limits with padding
        x_coords = [entity.position.x for entity in uav_manager.get_all() + task_manager.get_all()]
        y_coords = [entity.position.y for entity in uav_manager.get_all() + task_manager.get_all()]
        
        if x_coords and y_coords:  # Check if lists are not empty
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding (10% of range)
            x_padding = 0.1 * (x_max - x_min)
            y_padding = 0.1 * (y_max - y_min)
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Add a professional title with coalition information
        assigned_tasks = len([t for t, c in self.task2coalition.items() 
                             if t != self.free_uav_task_id and c])
        total_tasks = len(task_manager) - 1  # Subtract free UAV task
        
        plt.title(
            f"UAV Task Assignment Map ({assigned_tasks}/{total_tasks} Tasks Assigned)",
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        # Add timestamp and border
        plt.figtext(
            0.01, 0.01, 
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=8, 
            color='gray'
        )
        
        # Add border to make the plot more professional
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('#cccccc')
            
        # Adjust layout to prevent clipping
        plt.tight_layout()
        
        # Save with high quality
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
            
        plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    pass
