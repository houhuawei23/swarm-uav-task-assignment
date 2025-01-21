import numpy as np


class UAV:
    """Represents an Unmanned Aerial Vehicle (UAV) with its attributes and capabilities.

    uav.info = (Resource, Position, Value, Max Speed)

    Attributes:
        id (int): Unique identifier for the UAV.
        resources (list or np.ndarray): A vector representing the resources available on the UAV.
        position (tuple or list): A 3D coordinate (x, y, z) representing the UAV's current position.
        value (float): The intrinsic value or importance of the UAV.
        max_speed (float): The maximum speed at which the UAV can travel.
    """

    def __init__(self, id, resources, position, value, max_speed):
        self.id: int = id
        self.resources: np.ndarray = np.array(resources)  # 资源向量
        self.position: np.ndarray = np.array(position)  # 位置信息 (x, y, z)
        self.value: float = value  # 无人机价值
        self.max_speed: float = max_speed  # 最大速度

    # def __repr__(self):
    #     return f"UAV(id={self.id}, resources={self.resources}, position={self.position}, value={self.value}, max_speed={self.max_speed})"
    # def __str__(self):
    #     return f"u{self.id}"
    def __repr__(self):
        return f"u{self.id}"

    def __str__(self):
        return f"u{self.id}, re={self.resources}, pos={self.position}, val={self.value}, ms={self.max_speed}"


from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


class UAVManager:
    """Manages a fleet of UAVs and provides methods for UAV coordination and management."""

    def __init__(self, uavs: List[UAV] = None):
        """Initialize the UAVManager with an optional list of UAVs."""
        self.uavs: Dict[int, UAV] = {uav.id: uav for uav in uavs}
        # self.uavs = {}
        # if uavs:
        #     for uav in uavs:
        #         self.uavs[uav.id] = uav

    def add_uav(self, uav: UAV):
        """Add a UAV to the manager."""
        if uav.id in self.uavs:
            raise ValueError(f"UAV with ID {uav.id} already exists.")
        self.uavs[uav.id] = uav

    def delete_uav_by_id(self, uav_id: int):
        """Remove a UAV by its ID."""
        if uav_id not in self.uavs:
            raise KeyError(f"UAV with ID {uav_id} not found.")
        del self.uavs[uav_id]

    def get_uav_by_id(self, uav_id: int) -> Optional[UAV]:
        """Retrieve a UAV by its ID."""
        return self.uavs.get(uav_id)

    def get_all_uavs(self) -> List[UAV]:
        """Get a list of all managed UAVs."""
        return list(self.uavs.values())

    def get_uav_ids(self) -> List[int]:
        """Get a list of all UAV IDs in the manager."""
        return list(self.uavs.keys())

    def update_uav(self, uav: UAV):
        """Update an existing UAV's information."""
        if uav.id not in self.uavs:
            raise KeyError(f"UAV with ID {uav.id} not found.")
        self.uavs[uav.id] = uav

    def clear_uavs(self):
        """Remove all UAVs from the manager."""
        self.uavs.clear()

    def nums(self):
        return len(self.uavs)

    def __contains__(self, uav_id: int) -> bool:
        """Check if a UAV with the given ID exists in the manager."""
        return uav_id in self.uavs

    def __iter__(self):
        return iter(self.uavs.values())

    def __len__(self):
        return len(self.uavs)

    def __repr__(self):
        return f"UAVManager(uavs={list(self.uavs.values())})"

    def __str__(self):
        return f"UAVManager managing {len(self.uavs)} UAVs"

    def plot_distribution(self, figsize=(16, 8)):
        """Visualizes UAV distribution and states in a comprehensive dashboard."""
        if not self.uavs:
            print("No UAVs to visualize")
            return

        sns.set_theme(style="whitegrid")
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.labelsize"] = 12

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        # 3D Position Plot
        ax3d = fig.add_subplot(gs[:, 0:2], projection="3d")
        self._plot_3d_positions(ax3d, fig)

        # Resource Radar Plot
        ax_radar = fig.add_subplot(gs[0, 2], polar=True)
        self._plot_resource_radar(ax_radar)

        # Capability Bar Plot
        ax_bar = fig.add_subplot(gs[1, 2])
        self._plot_capabilities(ax_bar)

        plt.suptitle("UAV Fleet Status Dashboard", y=1.02, fontsize=16)
        plt.show()

    def _plot_3d_positions(self, ax, fig):
        """Helper function for 3D position visualization"""
        positions = np.array([uav.position for uav in self.uavs.values()])
        values = [uav.value for uav in self.uavs.values()]
        speeds = [uav.max_speed for uav in self.uavs.values()]
        ids = [f"UAV {uav.id}" for uav in self.uavs.values()]

        sc = ax.scatter3D(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=values,
            s=np.array(speeds) * 20,
            cmap="viridis",
            edgecolor="w",
            linewidth=0.5,
            depthshade=False,
        )

        # Annotate points
        for i, txt in enumerate(ids):
            ax.text(
                positions[i, 0],
                positions[i, 1],
                positions[i, 2],
                txt,
                fontsize=8,
                ha="center",
                va="bottom",
            )

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Altitude")
        ax.set_title("3D Position Distribution\n(Color=Value, Size=Max Speed)")
        fig.colorbar(sc, ax=ax, label="UAV Value", shrink=0.5)

    def _plot_resource_radar(self, ax):
        """Helper function for radar chart of resources"""
        resources = np.array([uav.resources for uav in self.uavs.values()])
        categories = [f"Resource {i+1}" for i in range(resources.shape[1])]
        labels = [f"UAV {uav.id}" for uav in self.uavs.values()]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)

        for i, (label, res) in enumerate(zip(labels, resources)):
            values = res.tolist()
            values += values[:1]
            ax.plot(
                angles, values, linewidth=2, linestyle="solid", label=label, alpha=0.7
            )
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title("Resource Distribution Radar Chart", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))

    def _plot_capabilities(self, ax):
        """Helper function for capability comparison"""
        ids = [uav.id for uav in self.uavs.values()]
        values = [uav.value for uav in self.uavs.values()]
        speeds = [uav.max_speed for uav in self.uavs.values()]

        x = np.arange(len(ids))
        width = 0.35

        ax.bar(x - width / 2, values, width, label="Value", color="skyblue")
        ax.bar(x + width / 2, speeds, width, label="Max Speed", color="salmon")

        ax.set_xticks(x)
        ax.set_xticklabels(ids)
        ax.set_xlabel("UAV ID")
        ax.set_title("Value and Speed Comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)


# Example usage:
if __name__ == "__main__":
    # Create UAVs
    uav1 = UAV(id=1, resources=[2, 3], position=[0, 0, 0], value=1.5, max_speed=10)
    uav2 = UAV(id=2, resources=[4, 1], position=[5, 5, 5], value=2.0, max_speed=15)

    # Initialize manager
    manager = UAVManager([uav1, uav2])

    # Add new UAV
    uav3 = UAV(id=3, resources=[5, 2], position=[10, 0, 5], value=1.8, max_speed=12)
    manager.add_uav(uav3)

    # Get UAV by ID
    print(manager.get_uav_by_id(2))

    # Update UAV
    uav2.position = [8, 8, 8]
    manager.update_uav(uav2)

    # Delete UAV
    manager.delete_uav_by_id(1)

    # List all UAVs
    print(manager.get_all_uavs())

    # Plot distribution
    manager.plot_distribution()

    # Clear all UAVs
    manager.clear_uavs()
    print(manager.get_all_uavs())
