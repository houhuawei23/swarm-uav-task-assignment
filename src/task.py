import numpy as np
from typing import List, Optional


class Task:
    """Represents a task that requires the cooperation of UAVs to be completed.

    task.info = (Resources, Position, Time Window, Threat Index)

    Attributes:
        id (int): Unique identifier for the task.
        resources (list or np.ndarray): A vector representing the resources required to complete the task.
        position (tuple or list): A 3D coordinate (x, y, z) representing the task's location.
        time_window (list): A time window [min_start, max_start] during which the task can be started.
        threat (float): A threat index representing the danger or risk associated with the task.
    """

    def __init__(self, id, resources, position, time_window, threat):
        self.id = id
        self.required_resources: np.ndarray = np.array(resources)  # 资源需求向量
        # 已满足资源向量
        # self.satisfied_resources: np.ndarray = np.zeros_like(resources)
        self.resources_nums: int = len(resources)  # 资源数量
        self.position: np.ndarray = np.array(position)  # 位置信息 (x, y, z)
        self.time_window = time_window  # 时间窗口 [min_start, max_start]
        self.threat = threat  # 威胁指数
        # # 资源权重向量，默认全部为1
        # self.resources_weights: np.ndarray = (
        #     np.ones(self.resources_nums) / self.resources_nums
        # )

    def get_resources_weights(self, task_obtained_resources=0):
        still_required_resources = self.required_resources - task_obtained_resources
        still_required_resources_pos = np.maximum(
            still_required_resources, 0
        )  # 将负值置为0
        if np.sum(still_required_resources_pos) == 0:
            return np.zeros_like(still_required_resources_pos)
        else:
            resources_weights = still_required_resources_pos / np.sum(
                still_required_resources_pos
            )
            return resources_weights

    def __repr__(self):
        return f"Task(id={self.id}, resources={self.required_resources}, position={self.position}, time_window={self.time_window}, threat_index={self.threat})"

    def __str__(self):
        return f"T{self.id}: re={self.required_resources}, pos={self.position}, tw={self.time_window}, thr={self.threat}"


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict


class TaskManager:
    """Manages a set of tasks and provides methods for task allocation and scheduling."""

    def __init__(self, tasks: List[Task] = None):
        """Initialize the TaskManager with a list of tasks."""
        self.tasks: Dict[int, Task] = {task.id: task for task in tasks}
        # if tasks:
        #     for task in tasks:
        #         self.tasks[task.id] = task

    def add_task(self, task: Task):
        """Add a task to the TaskManager."""
        if task.id in self.tasks:
            raise ValueError(f"Task with ID {task.id} already exists.")
        self.tasks[task.id] = task

    def delete_task_by_id(self, task_id: int):
        """Delete a task by its ID."""
        if task_id not in self.tasks:
            raise KeyError(f"Task with ID {task_id} not found.")
        del self.tasks[task_id]

    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """Get a task by its ID."""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """Get a list of all tasks."""
        # 返回一个包含所有任务的列表
        return list(self.tasks.values())

    def get_task_ids(self) -> List[int]:
        """Get a list of all task IDs."""
        return list(self.tasks.keys())

    def update_task(self, task: Task):
        """Update an existing task."""
        if task.id not in self.tasks:
            raise KeyError(f"Task with ID {task.id} not found.")
        self.tasks[task.id] = task

    def clear_tasks(self):
        """Clear all tasks from the TaskManager."""
        self.tasks.clear()

    def nums(self):
        return len(self.tasks)

    def __iter__(self):
        """Iterate over all tasks."""
        return iter(self.tasks.values())

    def __repr__(self):
        return f"TaskManager(tasks={list(self.tasks.values())})"

    def __str__(self):
        return f"TaskManager with {len(self.tasks)} tasks."

    def plot_distribution_beta(self, dimension="auto", figsize=(12, 8)):
        """Main visualization method with modular components."""
        if not self.tasks:
            print("No tasks to visualize")
            return

        self._setup_plot_style()
        positions = self._prepare_position_data()
        use_3d = self._determine_dimensionality(positions, dimension)
        fig, ax = self._create_figure_axes(figsize, use_3d)
        visual_data = self._extract_visualization_data()

        scatter = self._create_scatter_plot(ax, positions, visual_data, use_3d)
        self._configure_axes(ax, use_3d)
        self._add_annotations(
            ax, positions, visual_data["time_windows"], visual_data["ids"], use_3d
        )
        self._add_color_bar(fig, ax, scatter)
        self._add_size_legend(ax, visual_data["resources"], use_3d)
        self._finalize_plot(fig, len(self.tasks), use_3d)

    def _setup_plot_style(self):
        """Configure global plot styling."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(
            {"font.size": 10, "axes.labelsize": 12, "axes.titlesize": 14}
        )

    def _prepare_position_data(self):
        """Process and pad position data as needed."""
        positions = []
        for task in self.tasks.values():
            pos = task.position.copy()
            if len(pos) < 3:
                pos = np.pad(pos, (0, 3 - len(pos)), "constant")
            positions.append(pos)
        return np.array(positions)

    def _determine_dimensionality(self, positions, dimension):
        """Determine whether to use 3D visualization."""
        if dimension == "auto":
            return not np.all(positions[:, 2] == 0)
        return dimension == "3d"

    def _create_figure_axes(self, figsize, use_3d):
        """Create figure and axes with appropriate projection."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d") if use_3d else fig.add_subplot(111)
        return fig, ax

    def _extract_visualization_data(self):
        """Extract and package visualization parameters."""
        return {
            "threats": [t.threat for t in self.tasks.values()],
            "resources": [
                np.linalg.norm(t.required_resources) for t in self.tasks.values()
            ],
            "time_windows": [
                f"{t.time_window[0]}-{t.time_window[1]}" for t in self.tasks.values()
            ],
            "ids": [f"T{t.id}" for t in self.tasks.values()],
        }

    def _create_scatter_plot(self, ax, positions, visual_data, use_3d):
        """Create the main scatter plot visualization."""
        min_res, max_res = min(visual_data["resources"]), max(visual_data["resources"])
        sizes = np.interp(visual_data["resources"], (min_res, max_res), (100, 500))

        scatter_args = {
            "c": visual_data["threats"],
            "s": sizes,
            "cmap": "plasma",
            "alpha": 0.8,
            "edgecolor": "w",
            "linewidth": 0.5,
        }

        if use_3d:
            scatter_args["xs"] = positions[:, 0]
            scatter_args["ys"] = positions[:, 1]
            scatter_args["zs"] = positions[:, 2]
            return ax.scatter3D(**scatter_args)
        else:
            scatter_args["x"] = positions[:, 0]
            scatter_args["y"] = positions[:, 1]
            return ax.scatter(**scatter_args)

    def _configure_axes(self, ax, use_3d):
        """Configure axis labels and grid settings."""
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        if use_3d:
            ax.set_zlabel("Altitude")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, linestyle="--", alpha=0.5)

    def _add_annotations(self, ax, positions, time_windows, ids, use_3d):
        """Add text annotations to each task point."""
        for i, pos in enumerate(positions):
            label = f"{ids[i]}\n({time_windows[i]})"
            text_args = {
                "x": pos[0],
                "y": pos[1],
                "s": label,
                "fontsize": 7,
                "ha": "center",
                "va": "bottom",
                "linespacing": 1.2,
            }
            if use_3d:
                text_args["z"] = pos[2]
                ax.text(**text_args)
            else:
                ax.text(**text_args)

    def _add_color_bar(self, fig, ax, scatter):
        """Add and configure the color bar."""
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Threat Level", rotation=270, labelpad=15)

    def _add_size_legend(self, ax, resources, use_3d):
        """Add resource size legend."""
        min_res, max_res = min(resources), max(resources)
        legend_sizes = [min_res, (min_res + max_res) / 2, max_res]
        legend_elements = [
            plt.scatter(
                [],
                [],
                s=np.interp(s, (min_res, max_res), (100, 500)),
                edgecolor="w",
                alpha=0.8,
                label=f"{s:.1f}",
            )
            for s in legend_sizes
        ]
        ax.legend(
            handles=legend_elements,
            title="Resource Norm",
            bbox_to_anchor=(1.05 if use_3d else 1.15, 0.8),
            loc="upper left",
        )

    def _finalize_plot(self, fig, task_count, use_3d):
        """Apply final layout adjustments and titles."""
        plt.title(
            f"Task Distribution Visualization\n"
            f"Total Tasks: {task_count} | Display Mode: {'3D' if use_3d else '2D'}",
            pad=20,
        )
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Create some tasks
    task1 = Task(
        id=1,
        resources=[5, 2, 3],
        position=[10, 20, 0],
        time_window=[0, 100],
        threat=0.5,
    )
    task2 = Task(
        id=2,
        resources=[4, 5, 6],
        position=[40, 50, 0],
        time_window=[50, 150],
        threat=0.7,
    )

    # Initialize TaskManager with a list of tasks
    task_manager = TaskManager(tasks=[task1, task2])

    # Add a new task
    task3 = Task(
        id=3,
        resources=[7, 8, 9],
        position=[70, 80, 0],
        time_window=[100, 200],
        threat=0.9,
    )
    task_manager.add_task(task3)

    # Get a task by ID
    print(task_manager.get_task_by_id(2))

    # Delete a task by ID
    task_manager.delete_task_by_id(1)

    # Get all tasks
    print(task_manager.get_all_tasks())

    # Update a task
    task2.required_resources = [10, 11, 12]
    task_manager.update_task(task2)

    # Print all tasks
    print(task_manager.get_all_tasks())

    # Automatic detection (will show 3D because of task 3)
    task_manager.plot_distribution_beta("2d")

    # Clear all tasks
    task_manager.clear_tasks()
    print(task_manager.get_all_tasks())
