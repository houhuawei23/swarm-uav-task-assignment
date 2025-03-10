# test_base.py
import pytest
import numpy as np
from framework.base import Point, Entity, EntityManager


# Unit Tests for Point Class
def test_point_initialization():
    # Test valid initialization
    p = Point([1.0, 2.0, 3.0])
    assert np.array_equal(p.xyz, np.array([1.0, 2.0, 3.0]))

    # Test invalid initialization (wrong length)
    with pytest.raises(AssertionError):
        Point([1.0, 2.0])

    # Test invalid initialization (wrong type)
    with pytest.raises(TypeError):
        Point("invalid")


def test_point_properties():
    p = Point([1.0, 2.0, 3.0])
    assert p.x == 1.0
    assert p.y == 2.0
    assert p.z == 3.0


def test_point_operations():
    p1 = Point([1.0, 2.0, 3.0])
    p2 = Point([4.0, 5.0, 6.0])

    # Test addition
    p3 = p1 + p2
    assert np.array_equal(p3.xyz, np.array([5.0, 7.0, 9.0]))

    # Test subtraction
    p4 = p1 - p2
    assert np.array_equal(p4.xyz, np.array([-3.0, -3.0, -3.0]))

    # Test distance calculation
    distance = p1.distance_to(p2)
    assert pytest.approx(distance) == 5.196152422706632  # sqrt(3^2 + 3^2 + 3^2)


def test_point_tolist():
    p = Point([1.0, 2.0, 3.0])
    assert p.tolist() == [1.0, 2.0, 3.0]


# Unit Tests for Entity Class
def test_entity_initialization():
    p = Point([1.0, 2.0, 3.0])
    e = Entity(1, p)
    assert e.id == 1
    assert e.position == p


def test_entity_to_dict():
    p = Point([1.0, 2.0, 3.0])
    e = Entity(1, p)
    assert e.to_dict() == {"id": 1, "type": "Entity", "position": [1.0, 2.0, 3.0]}


def test_entity_from_dict():
    data = {"id": 1, "position": [1.0, 2.0, 3.0]}
    e = Entity.from_dict(data)
    assert e.id == 1
    assert np.array_equal(e.position.xyz, np.array([1.0, 2.0, 3.0]))


def test_entity_brief_info():
    p = Point([1.0, 2.0, 3.0])
    e = Entity(1, p)
    assert e.brief_info() == "Entity 1 at Point(xyz=array([1., 2., 3.]))"


# Unit Tests for EntityManager Class
def test_entity_manager_initialization():
    p = Point([1.0, 2.0, 3.0])
    e = Entity(1, p)
    manager = EntityManager([e])
    assert manager.size() == 1
    assert manager.get(1) == e


def test_entity_manager_add():
    manager = EntityManager()
    p = Point([1.0, 2.0, 3.0])
    e = Entity(1, p)
    manager.add(e)
    assert manager.size() == 1
    assert manager.get(1) == e

    # Test adding duplicate entity
    with pytest.raises(ValueError):
        manager.add(e)


def test_entity_manager_remove():
    p = Point([1.0, 2.0, 3.0])
    e = Entity(1, p)
    manager = EntityManager([e])
    manager.remove(1)
    assert manager.size() == 0

    # Test removing non-existent entity
    with pytest.raises(ValueError):
        manager.remove(1)


def test_entity_manager_get_all():
    p1 = Point([1.0, 2.0, 3.0])
    p2 = Point([4.0, 5.0, 6.0])
    e1 = Entity(1, p1)
    e2 = Entity(2, p2)
    manager = EntityManager([e1, e2])
    assert manager.get_all() == [e1, e2]


def test_entity_manager_random_one():
    p1 = Point([1.0, 2.0, 3.0])
    p2 = Point([4.0, 5.0, 6.0])
    e1 = Entity(1, p1)
    e2 = Entity(2, p2)
    manager = EntityManager([e1, e2])
    random_entity = manager.random_one()
    assert random_entity in [e1, e2]

    # Test random_one with no entities
    empty_manager = EntityManager()
    with pytest.raises(ValueError):
        empty_manager.random_one()


def test_entity_manager_with_multiple_entities():
    # Create points
    p1 = Point([1.0, 2.0, 3.0])
    p2 = Point([4.0, 5.0, 6.0])

    # Create entities
    e1 = Entity(1, p1)
    e2 = Entity(2, p2)

    # Create manager and add entities
    manager = EntityManager()
    manager.add(e1)
    manager.add(e2)

    # Verify manager state
    assert manager.size() == 2
    assert manager.get(1) == e1
    assert manager.get(2) == e2

    # Verify to_dict_list
    dict_list = manager.to_dict_list()
    assert dict_list == [
        {"id": 1, "type": "Entity", "position": [1.0, 2.0, 3.0]},
        {"id": 2, "type": "Entity", "position": [4.0, 5.0, 6.0]},
    ]

    # Remove an entity and verify
    manager.remove(1)
    assert manager.size() == 1
    assert manager.get(2) == e2


def test_entity_manager_plot_integration():
    import matplotlib.pyplot as plt

    # Create points and entities
    p1 = Point([1.0, 2.0, 3.0])
    p2 = Point([4.0, 5.0, 6.0])
    e1 = Entity(1, p1)
    e2 = Entity(2, p2)

    # Create manager and add entities
    manager = EntityManager([e1, e2])

    # Plot entities (visual verification is not automated)
    fig, ax = plt.subplots()
    manager.plot(ax)
    plt.close(fig)  # Close the plot to avoid displaying it during tests


def test_entity_from_dict_integration():
    # Create a dictionary representation of an entity
    data = {"id": 1, "position": [1.0, 2.0, 3.0]}

    # Create entity from dictionary
    e = Entity.from_dict(data)

    # Verify entity properties
    assert e.id == 1
    assert np.array_equal(e.position.xyz, np.array([1.0, 2.0, 3.0]))

    # Verify to_dict round-trip
    assert e.to_dict() == {"id": 1, "type": "Entity", "position": [1.0, 2.0, 3.0]}
