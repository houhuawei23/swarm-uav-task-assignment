from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


@dataclass(repr=True, eq=True)
class Point:
    xyz: np.ndarray = field(init=False)

    @property
    def x(self) -> float:
        return self.xyz[0]

    @property
    def y(self) -> float:
        return self.xyz[1]

    @property
    def z(self) -> float:
        return self.xyz[2]

    def __init__(self, xyz: List[float]):
        assert len(xyz) == 3
        self.xyz = np.array(xyz)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.xyz + other.xyz)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.xyz - other.xyz)

    def distance_to(self, other: "Point") -> float:
        return np.linalg.norm(self.xyz - other.xyz)

    def tolist(self) -> List[float]:
        return self.xyz.tolist()


@dataclass
class Entity:
    """Base class for all entities in the simulation."""

    id: int
    position: Point

    def __post_init__(self):
        self.position = Point(self.position)

    def type(self):
        return self.__class__.__name__

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "position": self.position.tolist(),
        }

    @classmethod
    def from_dict(self, data: Dict):
        self.id = data["id"]
        self.position = Point(data["position"])

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

    def __post_init__(self):
        self.entities = {entity.id: entity for entity in self.entities}

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

    def to_dict(self) -> Dict[int, Dict]:
        return [entity.to_dict() for entity in self.entities.values()]

    def __len__(self):
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities.values())

    def plot(self, ax: plt.Axes, **kwargs) -> None:
        plot_entities_on_axes(ax, self.entities.values(), **kwargs)


@dataclass(repr=True)
class HyperParams:
    resources_num: int  # 资源维度数
    map_shape: Tuple[int, int, int]  # 任务环境区域大小
    alpha: float  # 资源贡献权重
    beta: float  # 路径成本权重
    gamma: float  # 威胁权重
    mu: float # ui 加入 tj 无资源贡献时的惩罚 (path_cost)
    max_iter: int  # 最大迭代次数

if __name__ == "__main__":
    p1 = Point([1, 2, 3])
    p2 = Point([4, 5, 6])
    print(p1.distance_to(p2))  # 5.196152422706632

    e1 = Entity(1, [1, 2, 3])
    e2 = Entity(2, [4, 5, 6])
    em = EntityManager([e1, e2])
    print(em.get_all())

    fig, ax = plt.subplots()
    em.plot(ax, color="red", marker="s")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Entities")
    ax.grid(True)
    ax.legend()

    plt.show()
