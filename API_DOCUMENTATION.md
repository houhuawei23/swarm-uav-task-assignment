# Swarm UAV Task Assignment - API Documentation

## Overview

This repository provides a comprehensive framework for Multi-Robot Task Assignment (MRTA) problems, specifically designed for swarm UAV (Unmanned Aerial Vehicle) systems. The framework includes multiple solver algorithms, simulation capabilities, and evaluation tools.

## Table of Contents

- [Framework API](#framework-api)
  - [Core Classes](#core-classes)
  - [UAV Management](#uav-management)
  - [Task Management](#task-management)
  - [Coalition Management](#coalition-management)
  - [MRTA Solver Base](#mrta-solver-base)
- [Solver APIs](#solver-apis)
- [Main Scripts](#main-scripts)
- [Utility Functions](#utility-functions)
- [Usage Examples](#usage-examples)

---

## Framework API

### Core Classes

#### Point

A 3D point representation used throughout the framework.

```python
from framework.base import Point

class Point:
    """3D point with x, y, z coordinates."""
    
    def __init__(self, xyz: List[float] | np.ndarray | Point):
        """Initialize a point from coordinates."""
        
    @property
    def x(self) -> float:
        """Get x coordinate."""
        
    @property
    def y(self) -> float:
        """Get y coordinate."""
        
    @property
    def z(self) -> float:
        """Get z coordinate."""
        
    def distance_to(self, other: Point) -> float:
        """Calculate Euclidean distance to another point."""
        
    def direction_to(self, other: Point) -> np.ndarray:
        """Get unit direction vector to another point."""
        
    def copy(self) -> Point:
        """Create a copy of this point."""
```

**Usage Example:**
```python
# Create a point
point1 = Point([100.0, 200.0, 50.0])
point2 = Point([150.0, 250.0, 75.0])

# Calculate distance
distance = point1.distance_to(point2)
print(f"Distance: {distance}")

# Get direction
direction = point1.direction_to(point2)
```

#### Entity

Base class for all entities in the simulation (UAVs, Tasks).

```python
from framework.base import Entity

class Entity:
    """Base class for all entities in the simulation."""
    
    def __init__(self, id: int, position: Point | List[float] | np.ndarray):
        """Initialize entity with ID and position."""
        
    @classmethod
    def type_name(cls) -> str:
        """Get the class name."""
        
    def to_dict(self) -> Dict:
        """Convert entity to dictionary representation."""
        
    @classmethod
    def from_dict(cls, data: Dict) -> Entity:
        """Create entity from dictionary data."""
        
    def brief_info(self) -> str:
        """Get brief information string."""
```

#### HyperParams

Configuration parameters for MRTA solvers.

```python
from framework.base import HyperParams

@dataclass
class HyperParams:
    """Hyperparameters for MRTA algorithms."""
    
    max_iter: int = 100
    log_level: LogLevel = LogLevel.INFO
    tolerance: float = 1e-6
    random_seed: int = 42
```

### UAV Management

#### UAV

Represents an Unmanned Aerial Vehicle with its capabilities and state.

```python
from framework.uav import UAV, UAVState

class UAVState(Enum):
    """UAV operational states."""
    IDLE = "idle"
    TASK_ALLOCATING = "task_allocating"
    FLYING = "flying"
    TASK_EXECUTING = "task_executing"
    RETURNING = "returning"

class UAV(Entity):
    """Unmanned Aerial Vehicle with resources and capabilities."""
    
    def __init__(
        self,
        id: int,
        position: Point | List[float] | np.ndarray,
        resources: List[float],
        value: float,
        max_speed: float,
        mass: float = 1.0,
        fly_energy_per_time: float = 2.0,
        hover_energy_per_time: float = 1.5,
    ):
        """Initialize UAV with capabilities."""
        
    def can_execute_task(self, task: Task) -> bool:
        """Check if UAV can execute a task based on resources."""
        
    def calculate_travel_time(self, target_position: Point) -> float:
        """Calculate time to reach target position."""
        
    def calculate_energy_cost(self, target_position: Point, execution_time: float) -> float:
        """Calculate energy cost for task execution."""
        
    def to_dict(self) -> Dict:
        """Convert UAV to dictionary representation."""
        
    @classmethod
    def from_dict(cls, data: Dict) -> UAV:
        """Create UAV from dictionary data."""
```

**Usage Example:**
```python
# Create a UAV
uav = UAV(
    id=1,
    position=[100.0, 200.0, 50.0],
    resources=[10.0, 8.0, 12.0, 6.0, 9.0],
    value=100.0,
    max_speed=25.0,
    mass=12.0
)

# Check capabilities
print(f"UAV {uav.id} at position {uav.position}")
print(f"Resources: {uav.resources}")
print(f"Current state: {uav.state}")
```

#### UAVManager

Manages a collection of UAVs.

```python
from framework.uav import UAVManager

class UAVManager(EntityManager):
    """Manager for UAV collections."""
    
    def __init__(self, uav_list: List[UAV] = None):
        """Initialize with list of UAVs."""
        
    def get_idle_uavs(self) -> List[UAV]:
        """Get all idle UAVs."""
        
    def get_available_resources(self, uav_ids: List[int]) -> np.ndarray:
        """Calculate total available resources from UAV coalition."""
        
    def calculate_coalition_value(self, uav_ids: List[int]) -> float:
        """Calculate total value of UAV coalition."""
        
    def find_capable_uavs(self, required_resources: np.ndarray) -> List[UAV]:
        """Find UAVs capable of contributing to required resources."""
```

**Usage Example:**
```python
# Create UAV manager
uavs = [uav1, uav2, uav3]  # List of UAV objects
uav_manager = UAVManager(uavs)

# Get idle UAVs
idle_uavs = uav_manager.get_idle_uavs()
print(f"Found {len(idle_uavs)} idle UAVs")

# Calculate coalition resources
coalition_ids = [1, 2, 3]
total_resources = uav_manager.get_available_resources(coalition_ids)
```

### Task Management

#### Task

Represents a task that requires UAV cooperation to complete.

```python
from framework.task import Task, TaskState

class TaskState(Enum):
    """Task execution states."""
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"

class Task(Entity):
    """Task requiring UAV cooperation."""
    
    def __init__(
        self,
        id: int,
        position: Point,
        required_resources: List[float],
        time_window: List[float],
        threat: float,
        execution_time: float = 2.0,
    ):
        """Initialize task with requirements."""
        
    def is_feasible_with_resources(self, available_resources: np.ndarray) -> bool:
        """Check if task can be completed with given resources."""
        
    def is_within_time_window(self, current_time: float) -> bool:
        """Check if current time is within task's time window."""
        
    def calculate_resource_satisfaction(self, available_resources: np.ndarray) -> float:
        """Calculate how well available resources satisfy requirements."""
        
    def brief_info(self) -> str:
        """Get brief task information."""
```

**Usage Example:**
```python
# Create a task
task = Task(
    id=1,
    position=[300.0, 400.0, 0.0],
    required_resources=[15.0, 10.0, 8.0, 12.0, 6.0],
    time_window=[10.0, 50.0],
    threat=0.3,
    execution_time=3.0
)

# Check task properties
print(f"Task {task.id} requires resources: {task.required_resources}")
print(f"Time window: {task.time_window}")
print(f"Threat level: {task.threat}")
```

#### TaskManager

Manages a collection of tasks.

```python
from framework.task import TaskManager

class TaskManager(EntityManager):
    """Manager for task collections."""
    
    def __init__(self, task_list: List[Task] = None):
        """Initialize with list of tasks."""
        
    def get_unassigned_tasks(self) -> List[Task]:
        """Get all unassigned tasks."""
        
    def get_tasks_in_time_window(self, current_time: float) -> List[Task]:
        """Get tasks available at current time."""
        
    def calculate_total_required_resources(self) -> np.ndarray:
        """Calculate total resources required by all tasks."""
        
    def get_high_priority_tasks(self, threshold: float = 0.7) -> List[Task]:
        """Get tasks with low threat (high priority)."""
```

### Coalition Management

#### CoalitionManager

Manages UAV-Task assignments and coalitions.

```python
from framework.coalition_manager import CoalitionManager

class CoalitionManager:
    """Manages UAV coalitions and task assignments."""
    
    def __init__(self, uav_ids: List[int], task_ids: List[int]):
        """Initialize coalition manager."""
        
    def assign_coalition_to_task(self, task_id: int, uav_coalition: List[int]):
        """Assign a UAV coalition to a task."""
        
    def get_task_coalition(self, task_id: int) -> List[int]:
        """Get UAV coalition assigned to a task."""
        
    def get_uav_assignment(self, uav_id: int) -> int | None:
        """Get task assigned to a UAV."""
        
    def is_uav_available(self, uav_id: int) -> bool:
        """Check if UAV is available for assignment."""
        
    def remove_assignment(self, task_id: int):
        """Remove task assignment."""
        
    def get_completion_rate(self) -> float:
        """Calculate task completion rate."""
        
    def get_assignment_summary(self) -> Dict:
        """Get summary of current assignments."""
```

**Usage Example:**
```python
# Create coalition manager
uav_ids = [1, 2, 3, 4, 5]
task_ids = [1, 2, 3]
coalition_manager = CoalitionManager(uav_ids, task_ids)

# Assign coalition to task
coalition_manager.assign_coalition_to_task(task_id=1, uav_coalition=[1, 2, 3])

# Check assignments
assignment = coalition_manager.get_task_coalition(task_id=1)
print(f"Task 1 assigned to UAVs: {assignment}")

# Get completion rate
completion_rate = coalition_manager.get_completion_rate()
print(f"Completion rate: {completion_rate:.2f}")
```

### MRTA Solver Base

#### MRTASolver

Base class for all Multi-Robot Task Assignment solvers.

```python
from framework.mrta_solver import MRTASolver

class MRTASolver:
    """Base class for MRTA solvers."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        """Initialize solver with managers and parameters."""
        
    @classmethod
    def type_name(cls) -> str:
        """Get solver type name."""
        
    @staticmethod
    def uav_type() -> Type[UAV]:
        """Get UAV type used by this solver."""
        
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Run the allocation algorithm."""
        # To be implemented by subclasses
        raise NotImplementedError("This method should be implemented by the subclass.")
```

---

## Solver APIs

The framework includes multiple solver implementations for different MRTA approaches:

### Coalition Formation Game Solvers

#### ChinaScience2024_CoalitionFormationGame

Implementation of the algorithm from CSCI 2024 paper.

```python
from solvers.csci2024 import ChinaScience2024_CoalitionFormationGame

class ChinaScience2024_CoalitionFormationGame(MRTASolver):
    """Coalition Formation Game solver from CSCI 2024."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Run the CFG allocation algorithm."""
```

#### IROS2024_CoalitionFormationGame

Implementation of the algorithm from IROS 2024 paper.

```python
from solvers.iros2024 import IROS2024_CoalitionFormationGame

class IROS2024_CoalitionFormationGame(MRTASolver):
    """Coalition Formation Game solver from IROS 2024."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Run the IROS 2024 CFG algorithm."""
```

#### ICRA2024_CoalitionFormationGame

Implementation of the algorithm from ICRA 2024 paper.

```python
from solvers.icra2024 import ICRA2024_CoalitionFormationGame

class ICRA2024_CoalitionFormationGame(MRTASolver):
    """Coalition Formation Game solver from ICRA 2024."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Run the ICRA 2024 CFG algorithm."""
```

### Optimization-Based Solvers

#### MILPSolver

Mixed-Integer Linear Programming solver.

```python
from solvers.milp_solver import MILPSolver

class MILPSolver(MRTASolver):
    """MILP-based task assignment solver."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Solve using MILP optimization."""
```

#### NLPSolverScipy

Non-Linear Programming solver using SciPy.

```python
from solvers.nlp_solver import NLPSolverScipy

class NLPSolverScipy(MRTASolver):
    """NLP solver using SciPy optimization."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Solve using NLP optimization."""
```

### Auction-Based Solvers

#### AuctionBiddingSolver

Auction-based task assignment solver.

```python
from solvers.auction_solver import AuctionBiddingSolver

class AuctionBiddingSolver(MRTASolver):
    """Auction-based task assignment solver."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Run auction-based allocation."""
```

### Distributed Solvers

#### DistributedSolver

Distributed multi-agent solver.

```python
from solvers.distributed_solver import DistributedSolver

class DistributedSolver(MRTASolver):
    """Distributed multi-agent task assignment solver."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Run distributed allocation algorithm."""
```

### Enumeration Solver

#### EnumerationSolver

Exhaustive search solver for small problems.

```python
from solvers.enum_solver import EnumerationSolver

class EnumerationSolver(MRTASolver):
    """Enumeration-based solver for small problems."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Run exhaustive search allocation."""
```

---

## Main Scripts

### main.py

Main entry point for running tests and experiments.

```bash
# Test algorithms with different parameters
python main.py test --test_case uav_num --uav_nums 10 20 30 --task_nums 5 10 15 --choices csci2024 iros2024

# Plot results
python main.py plot --file_path results.json --x uav_num --labels "Completion Rate" --choices csci2024

# Dynamic simulation
python main.py dynamic --test_case dynamic_test --uav_nums 20 --task_nums 10
```

**Command Line Arguments:**

#### Test Command
- `--test_case`: Test case type (`uav_num`, `task_num`, `hyper_params.xxx`, or file path)
- `--uav_nums`: List of UAV numbers to test (default: [40])
- `--task_nums`: List of task numbers to test (default: [20])
- `--choices`: Solver algorithms to test
- `--timeout`: Timeout for each test in seconds
- `--random_test_times`: Number of random test iterations
- `--show`: Show plots during execution
- `--save_dir`: Directory to save results
- `--sim`: Enable simulation mode

#### Plot Command
- `--file_path`: Path to results file
- `--x`: X-axis variable for plotting
- `--labels`: Labels for plot data
- `--choices`: Solvers to include in plot
- `--show`: Show plots
- `--save_dir`: Directory to save plots

### gen.py

Test case generation script.

```bash
# Generate test cases for different scenarios
python gen.py --output_path test_cases/custom_case.json --scenario iros2024
```

**Usage Example:**
```python
from gen import gen_iros2024_case, gen_icra2024_case

# Generate IROS 2024 test case
gen_iros2024_case("test_cases/iros_case.json")

# Generate ICRA 2024 test case
gen_icra2024_case("test_cases/icra_case.json")
```

---

## Utility Functions

### framework.utils

#### Evaluation Functions

```python
from framework.utils import EvalResult, calculate_obtained_resources

@dataclass
class EvalResult:
    """Evaluation results for solver performance."""
    completion_rate: float = 0.0
    resource_use_rate: float = 0.0
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_exploss: float = 0.0
    elapsed_time: float = 0.0
    
    def format_print(self):
        """Print formatted evaluation results."""
        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""

def calculate_obtained_resources(
    coalition: List[int], 
    uav_manager: UAVManager, 
    resources_num: int
) -> np.ndarray:
    """Calculate total resources available from UAV coalition."""

def get_resources_weights(
    required_resources: np.ndarray, 
    task_obtained_resources: np.ndarray
) -> np.ndarray:
    """Calculate resource weights based on remaining requirements."""
```

#### Formatting Functions

```python
def format_with_prettier(json_string: str) -> str:
    """Format JSON string using Prettier."""

def calculate_map_shape_on_mana(
    uav_manager: UAVManager, 
    task_manager: TaskManager
) -> List[float]:
    """Calculate map boundaries from entities."""
```

### solvers.utils

#### Activation Functions

```python
from solvers.utils import Activations

class Activations:
    """Activation functions for neural network components."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
```

---

## Usage Examples

### Basic Framework Usage

```python
from framework import UAV, Task, UAVManager, TaskManager, CoalitionManager, HyperParams
from solvers import ChinaScience2024_CoalitionFormationGame
import numpy as np

# Create UAVs
uavs = [
    UAV(id=1, position=[0, 0, 0], resources=[10, 8, 12], value=100, max_speed=25),
    UAV(id=2, position=[50, 50, 0], resources=[8, 10, 9], value=90, max_speed=30),
    UAV(id=3, position=[100, 0, 0], resources=[12, 6, 15], value=110, max_speed=20),
]

# Create tasks
tasks = [
    Task(id=1, position=[200, 100, 0], required_resources=[15, 10, 8], 
         time_window=[0, 30], threat=0.2),
    Task(id=2, position=[150, 200, 0], required_resources=[12, 15, 12], 
         time_window=[10, 40], threat=0.4),
]

# Create managers
uav_manager = UAVManager(uavs)
task_manager = TaskManager(tasks)
coalition_manager = CoalitionManager(
    uav_ids=[uav.id for uav in uavs],
    task_ids=[task.id for task in tasks]
)

# Set hyperparameters
hyper_params = HyperParams(max_iter=100, log_level=LogLevel.INFO)

# Create and run solver
solver = ChinaScience2024_CoalitionFormationGame(
    uav_manager, task_manager, coalition_manager, hyper_params
)

# Run allocation
result_coalition_manager = solver.run_allocate()

# Check results
completion_rate = result_coalition_manager.get_completion_rate()
print(f"Task completion rate: {completion_rate:.2f}")

# Get assignments
for task_id in task_manager.get_ids():
    coalition = result_coalition_manager.get_task_coalition(task_id)
    print(f"Task {task_id} assigned to UAVs: {coalition}")
```

### Running Experiments

```python
import subprocess
import json

# Run comparative experiment
result = subprocess.run([
    "python", "main.py", "test",
    "--test_case", "uav_num",
    "--uav_nums", "10", "20", "30", "40",
    "--task_nums", "5", "10", "15",
    "--choices", "csci2024", "iros2024", "icra2024",
    "--random_test_times", "5",
    "--save_dir", "results/experiment_1"
], capture_output=True, text=True)

if result.returncode == 0:
    print("Experiment completed successfully")
else:
    print(f"Error: {result.stderr}")
```

### Custom Solver Implementation

```python
from framework.mrta_solver import MRTASolver
from framework.coalition_manager import CoalitionManager

class CustomSolver(MRTASolver):
    """Custom MRTA solver implementation."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager:
        """Custom allocation algorithm."""
        
        # Initialize
        self.coalition_manager.clear_all_assignments()
        
        # Get unassigned tasks
        unassigned_tasks = self.task_manager.get_unassigned_tasks()
        
        for task in unassigned_tasks:
            # Find best coalition for this task
            best_coalition = self._find_best_coalition(task)
            
            if best_coalition:
                self.coalition_manager.assign_coalition_to_task(
                    task.id, best_coalition
                )
        
        return self.coalition_manager
    
    def _find_best_coalition(self, task):
        """Find the best UAV coalition for a task."""
        # Custom logic for coalition formation
        available_uavs = self.uav_manager.get_idle_uavs()
        
        # Simple greedy approach: select UAVs with highest resource match
        selected_uavs = []
        remaining_resources = task.required_resources.copy()
        
        for uav in available_uavs:
            if np.any(uav.resources > 0) and np.any(remaining_resources > 0):
                selected_uavs.append(uav.id)
                remaining_resources -= uav.resources
                remaining_resources = np.maximum(remaining_resources, 0)
                
                # Stop if requirements are satisfied
                if np.all(remaining_resources <= 0):
                    break
        
        return selected_uavs if selected_uavs else None

# Usage
solver = CustomSolver(uav_manager, task_manager, coalition_manager, hyper_params)
result = solver.run_allocate()
```

### Simulation and Visualization

```python
from framework.sim import SimulationEnv
from framework.test import TestFramework

# Create simulation environment
sim_env = SimulationEnv(uav_manager, task_manager)

# Run simulation with visualization
test_framework = TestFramework()
results = test_framework.run_test_with_simulation(
    solver_class=ChinaScience2024_CoalitionFormationGame,
    test_case_path="test_cases/example.json",
    show_animation=True
)

# Evaluate results
eval_result = results['eval_result']
eval_result.format_print()
```

### Loading and Saving Test Cases

```python
import json
from framework.uav import UAV
from framework.task import Task

# Load test case from JSON
with open("test_cases/example.json", "r") as f:
    data = json.load(f)

# Create entities from data
uavs = [UAV.from_dict(uav_data) for uav_data in data["uavs"]]
tasks = [Task.from_dict(task_data) for task_data in data["tasks"]]

# Save results
results = {
    "solver": "csci2024",
    "completion_rate": 0.85,
    "assignments": coalition_manager.get_assignment_summary()
}

with open("results/experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Configuration and Parameters

### HyperParameters

Different solvers may use specific hyperparameters:

```python
# Coalition Formation Game parameters
cfg_params = HyperParams(
    max_iter=50,
    tolerance=1e-4,
    coalition_threshold=0.7,
    resource_weight=0.6,
    distance_weight=0.3,
    threat_weight=0.1
)

# Optimization solver parameters  
opt_params = HyperParams(
    max_iter=1000,
    tolerance=1e-6,
    solver_method="SLSQP",
    constraint_tolerance=1e-3
)

# Auction solver parameters
auction_params = HyperParams(
    max_iter=20,
    bid_increment=0.1,
    reserve_price=10.0,
    auction_rounds=5
)
```

### Test Case Generation Parameters

```python
from framework.uav import UAVGenParams
from framework.task import TaskGenParams

# UAV generation parameters
uav_params = UAVGenParams(
    region_ranges=[(0, 1000), (0, 1000), (0, 100)],
    resources_num=5,
    resources_range=(5, 15),
    value_range=(50, 150),
    speed_range=(15, 35),
    uav_mass_range=(8, 20),
    fly_energy_per_time_range=(1, 4),
    hover_energy_per_time_range=(0.5, 2),
)

# Task generation parameters  
task_params = TaskGenParams(
    region_ranges=[(100, 900), (100, 900), (0, 50)],
    resources_num=5,
    required_resources_range=(20, 50),
    time_window_ranges=[(0, 20), (60, 100)],
    threat_range=(0.1, 0.8),
    execution_time_range=(1, 5)
)
```

This comprehensive API documentation provides detailed information about all public APIs, functions, and components in the Swarm UAV Task Assignment framework, along with practical usage examples for each major component.