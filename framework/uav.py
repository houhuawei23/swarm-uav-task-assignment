from typing import List, Dict
from dataclasses import dataclass, field

import random
import numpy as np

from .base import Point, Entity, EntityManager


@dataclass(init=True, repr=True)
class UAV(Entity):
    """Represents an Unmanned Aerial Vehicle (UAV) with its attributes and capabilities.

    uav.info = (Resource, Position, Value, Max Speed)

    Attributes:
        id (int): Unique identifier for the UAV.
        resources (list or np.ndarray): A vector representing the resources available on the UAV.
        position (tuple or list): A 3D coordinate (x, y, z) representing the UAV's current position.
        value (float): The intrinsic value or importance of the UAV.
        max_speed (float): The maximum speed at which the UAV can travel.
    """

    resources: np.ndarray  # 资源向量
    value: float  # 无人机价值
    max_speed: float  # 最大速度

    mass: float  # 无人机重量
    fly_energy_per_time: float  # 飞行能耗 per time, per mass
    hover_energy_per_time: float  # 悬停能耗 per time, per mass

    def __post_init__(self):
        super().__post_init__()
        self.resources = np.array(self.resources)

    def to_dict(self):
        return {
            "id": self.id,
            "resources": self.resources.tolist(),
            "position": self.position.tolist(),
            "value": self.value,
            "max_speed": self.max_speed,
            "mass": self.mass,
            "fly_energy_per_time": self.fly_energy_per_time,
            "hover_energy_per_time": self.hover_energy_per_time,
        }

    @classmethod
    # 从字典中创建一个对象
    def from_dict(cls, data: Dict):
        return cls(
            id=data["id"],
            resources=data["resources"],
            position=data["position"],
            value=data["value"],
            max_speed=data["max_speed"],
            mass=data["mass"] if data.get("mass") is not None else 1.0,
            fly_energy_per_time=data["fly_energy_per_time"]
            if data.get("fly_energy_per_time") is not None
            else random.uniform(1, 3),
            hover_energy_per_time=data["hover_energy_per_time"]
            if data.get("hover_energy_per_time") is not None
            else random.uniform(1, 3),
        )

    def brief_info(self) -> str:
        return f"U_{self.id}(re={self.resources}, val={self.value}, spd={self.max_speed})"

    def cal_fly_energy(self, target_pos: Point) -> float:
        return (
            self.mass
            * (self.position.distance_to(target_pos) / self.max_speed)
            * self.fly_energy_per_time
        )

    def cal_hover_energy(self, hover_time: float) -> float:
        return self.mass * hover_time * self.hover_energy_per_time


from typing import Type


@dataclass
class UAVManager(EntityManager):
    def __init__(self, uavs: List[UAV] = []):
        super().__init__(uavs)

    def get(self, id: int) -> UAV:
        return super().get(id)

    def random_one(self) -> UAV:
        return super().random_one()

    @classmethod
    def from_dict(cls, data: List[Dict], UAVType: Type[UAV] = UAV):
        uavs = [UAVType.from_dict(task_data) for task_data in data]
        return cls(uavs)

    def format_print(self):
        print(f"UAVManager with {len(self)} uavs.")
        for uav in self.get_all():
            print(f"  {uav}")


# Example usage:
if __name__ == "__main__":
    # Create UAVs
    uav1 = UAV(id=1, resources=[2, 3], position=[0, 0, 0], value=1.5, max_speed=10)
    uav2 = UAV(id=2, resources=[4, 1], position=[5, 5, 5], value=2.0, max_speed=15)

    # Initialize manager
    manager = UAVManager([uav1, uav2])

    # Add new UAV
    uav3 = UAV(id=3, resources=[5, 2], position=[10, 0, 5], value=1.8, max_speed=12)
    # manager.add_uav(uav3)

    # # Get UAV by ID
    # print(manager.get_uav_by_id(2))

    # # Update UAV
    # uav2.position = [8, 8, 8]
    # manager.update_uav(uav2)

    # # Delete UAV
    # manager.delete_uav_by_id(1)

    # # List all UAVs
    # print(manager.get_all_uavs())

    # # Plot distribution
    # # manager.plot_distribution()

    # # Clear all UAVs
    # manager.clear_uavs()
    # print(manager.get_all_uavs())
