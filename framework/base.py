from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import random


@dataclass(repr=True, eq=True)
class Point:
    xyz: np.ndarray

    def __init__(self, xyz: List[float] | np.ndarray | "Point"):
        if isinstance(xyz, Point):
            self.xyz = xyz.xyz
        elif isinstance(xyz, (list, np.ndarray)):
            assert len(xyz) == 3
            self.xyz = np.array(xyz)
        else:
            raise TypeError("xyz must be a list, numpy array, or Point")

    @property
    def x(self) -> float:
        return self.xyz[0]

    @property
    def y(self) -> float:
        return self.xyz[1]

    @property
    def z(self) -> float:
        return self.xyz[2]

    def __eq__(self, other: "Point") -> bool:
        return np.all(self.xyz == other.xyz)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.xyz + other.xyz)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.xyz - other.xyz)

    def distance_to(self, other: "Point") -> float:
        return np.linalg.norm(self.xyz - other.xyz)

    def direction_to(self, other: "Point") -> np.ndarray:
        delta = other.xyz - self.xyz
        return delta / np.linalg.norm(delta)

    def tolist(self) -> List[float]:
        return self.xyz.tolist()


@dataclass
class Entity:
    """Base class for all entities in the simulation."""

    id: int
    position: Point

    def __init__(self, id: int, position: Point | List[float] | np.ndarray):
        self.id = id
        self.position = Point(position)
        if not isinstance(self.position, Point):
            raise TypeError("position must be a Point")
        # self.position = Point(position) if isinstance(position, (list, np.ndarray)) else position

    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError("id must be an integer")
        if not isinstance(self.position, Point):
            raise TypeError("position must be a Point")

    def type(self):
        return self.__class__.__name__

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type(),
            "position": self.position.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Entity":
        return cls(data["id"], data["position"])

    def brief_info(self) -> str:
        return f"{self.type()} {self.id} at {self.position}"


def plot_entities_on_axes(ax: plt.Axes, entities: List[Entity], **kwargs) -> None:
    text_delta = 0.2  # 文本偏移量
    for entity in entities:
        x, y, z = entity.position.xyz
        ax.scatter(x, y, label=f"{entity.type()} {entity.id}", **kwargs)
        ax.text(x, y + text_delta, s=entity.brief_info(), ha="center")


@dataclass
class EntityManager:
    # entities: List[Entity] = field(default_factory=list)
    entities: Dict[int, Entity] = field(default_factory=dict)

    def __init__(self, entity_list: List[Entity] = []):
        if entity_list is None:
            self.entities = {}
        else:
            self.entities = {entity.id: entity for entity in entity_list}

    def add(self, entity: Entity):
        if entity.id in self.entities:
            raise ValueError(f"Entity with id {entity.id} already exists.")
        self.entities[entity.id] = entity

    def remove(self, entity_id: int):
        if entity_id not in self.entities:
            raise ValueError(f"Entity with id {entity_id} does not exist.")
        del self.entities[entity_id]

    def get(self, entity_id: int) -> Entity:
        if entity_id not in self.entities:
            raise ValueError(f"Entity with id {entity_id} does not exist.")
        return self.entities[entity_id]

    def get_all(self) -> List[Entity]:
        return list(self.entities.values())

    def get_ids(self) -> List[int]:
        return list(self.entities.keys())

    def to_dict_list(self) -> List[Dict[int, Dict]]:
        return [entity.to_dict() for entity in self.entities.values()]

    def size(self) -> int:
        return len(self.entities)

    def __len__(self):
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities.values())

    def plot(self, ax: plt.Axes, **kwargs) -> None:
        plot_entities_on_axes(ax, self.entities.values(), **kwargs)

    def random_one(self) -> Entity:
        if not self.entities:
            raise ValueError("No entities in the manager.")
        # return random.sample(list(self.entities.values()), 1)[0]
        # which one is better?
        return random.choice(list(self.entities.values()))

    def brief_info(self):
        return str(self.get_ids())


@dataclass(repr=True)
class HyperParams:
    resources_num: int  # 资源维度数
    map_shape: Tuple[int, int, int]  # 任务环境区域大小
    alpha: float  # 资源贡献权重
    beta: float  # 路径成本权重
    gamma: float  # 威胁权重
    mu: float  # ui 加入 tj 无资源贡献时的惩罚 (path_cost)
    max_iter: int  # 最大迭代次数


@dataclass
class GenParams:
    region_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0, 100), (0, 100), (0, 100)]
    )
    resources_num: int = field(default=3)


if __name__ == "__main__":
    pass
