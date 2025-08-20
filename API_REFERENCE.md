# API Reference

## Overview

This document provides a complete API reference for the Swarm UAV Task Assignment framework. All public classes, methods, functions, and their parameters are documented here.

## Table of Contents

- [Framework Module](#framework-module)
- [Solvers Module](#solvers-module)
- [Main Scripts](#main-scripts)
- [Constants and Enums](#constants-and-enums)
- [Type Definitions](#type-definitions)

---

## Framework Module

### framework.base

#### Classes

##### Point
```python
class Point:
    """3D point representation."""
    
    def __init__(self, xyz: List[float] | np.ndarray | Point)
    
    @property
    def x(self) -> float
    
    @property
    def y(self) -> float
    
    @property
    def z(self) -> float
    
    def distance_to(self, other: Point) -> float
    
    def direction_to(self, other: Point) -> np.ndarray
    
    def tolist(self) -> List[float]
    
    def copy(self) -> Point
    
    def __add__(self, other: Point) -> Point
    
    def __sub__(self, other: Point) -> Point
    
    def __eq__(self, other: Point) -> bool
```

##### Entity
```python
class Entity:
    """Base class for all entities (UAVs, Tasks)."""
    
    def __init__(self, id: int, position: Point | List[float] | np.ndarray)
    
    @classmethod
    def type_name(cls) -> str
    
    def to_dict(self) -> Dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> Entity
    
    def brief_info(self) -> str
```

##### EntityManager
```python
class EntityManager:
    """Base manager for entity collections."""
    
    def __init__(self, entity_list: List[Entity] = None)
    
    def add(self, entity: Entity) -> None
    
    def remove(self, entity_id: int) -> None
    
    def get(self, entity_id: int) -> Entity
    
    def get_all(self) -> List[Entity]
    
    def get_ids(self) -> List[int]
    
    def size(self) -> int
    
    def clear(self) -> None
```

##### HyperParams
```python
@dataclass
class HyperParams:
    """Hyperparameters for MRTA algorithms."""
    
    max_iter: int = 100
    log_level: LogLevel = LogLevel.INFO
    tolerance: float = 1e-6
    random_seed: int = 42
    alpha: float = 0.6        # Resource weight
    beta: float = 0.3         # Distance weight
    gamma: float = 0.1        # Threat weight
```

##### GenParams
```python
@dataclass
class GenParams:
    """Base class for generation parameters."""
    
    region_ranges: List[Tuple[float, float]] = field(default_factory=list)
    resources_num: int = 5
```

#### Enums

##### LogLevel
```python
class LogLevel(IntEnum):
    SILENCE = 0
    INFO = 1
    DEBUG = 2
```

### framework.uav

#### Classes

##### UAV
```python
class UAV(Entity):
    """Unmanned Aerial Vehicle representation."""
    
    def __init__(
        self,
        id: int,
        position: Point | List[float] | np.ndarray,
        resources: List[float],
        value: float,
        max_speed: float,
        mass: float = 1.0,
        fly_energy_per_time: float = 2.0,
        hover_energy_per_time: float = 1.5
    )
    
    def can_execute_task(self, task: Task) -> bool
    
    def calculate_travel_time(self, target_position: Point) -> float
    
    def calculate_energy_cost(self, target_position: Point, execution_time: float) -> float
    
    def to_dict(self) -> Dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> UAV
    
    def brief_info(self) -> str
    
    # Properties
    @property
    def state(self) -> UAVState
    
    @property
    def resources(self) -> np.ndarray
    
    @property
    def value(self) -> float
    
    @property
    def max_speed(self) -> float
    
    @property
    def mass(self) -> float
    
    @property
    def fly_energy_per_time(self) -> float
    
    @property
    def hover_energy_per_time(self) -> float
```

##### UAVManager
```python
class UAVManager(EntityManager):
    """Manager for UAV collections."""
    
    def __init__(self, uav_list: List[UAV] = None)
    
    def get_idle_uavs(self) -> List[UAV]
    
    def get_available_resources(self, uav_ids: List[int]) -> np.ndarray
    
    def calculate_coalition_value(self, uav_ids: List[int]) -> float
    
    def find_capable_uavs(self, required_resources: np.ndarray) -> List[UAV]
    
    def get_uavs_in_range(self, position: Point, max_distance: float) -> List[UAV]
    
    def update_uav_states(self, assignments: Dict[int, int]) -> None
```

##### UAVGenParams
```python
@dataclass
class UAVGenParams(GenParams):
    """Parameters for UAV generation."""
    
    resources_range: Tuple[float, float] = (5.0, 15.0)
    value_range: Tuple[float, float] = (50.0, 150.0)
    speed_range: Tuple[float, float] = (15.0, 35.0)
    uav_mass_range: Tuple[float, float] = (8.0, 20.0)
    fly_energy_per_time_range: Tuple[float, float] = (1.0, 4.0)
    hover_energy_per_time_range: Tuple[float, float] = (0.5, 2.0)
    comm_bandwidth_range: Tuple[float, float] = (5.0, 15.0)
    trans_power_range: Tuple[float, float] = (20.0, 30.0)
```

#### Enums

##### UAVState
```python
class UAVState(Enum):
    IDLE = "idle"
    TASK_ALLOCATING = "task_allocating"
    FLYING = "flying"
    TASK_EXECUTING = "task_executing"
    RETURNING = "returning"
```

#### Functions

```python
def generate_uav_dict_list(
    uav_num: int,
    params: UAVGenParams,
    seed: int = None
) -> List[Dict]
```

### framework.task

#### Classes

##### Task
```python
class Task(Entity):
    """Task representation requiring UAV cooperation."""
    
    def __init__(
        self,
        id: int,
        position: Point,
        required_resources: List[float],
        time_window: List[float],
        threat: float,
        execution_time: float = 2.0
    )
    
    def is_feasible_with_resources(self, available_resources: np.ndarray) -> bool
    
    def is_within_time_window(self, current_time: float) -> bool
    
    def calculate_resource_satisfaction(self, available_resources: np.ndarray) -> float
    
    def is_zero_task(self) -> bool
    
    def to_dict(self) -> Dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> Task
    
    def brief_info(self) -> str
    
    # Properties
    @property
    def state(self) -> TaskState
    
    @property
    def required_resources(self) -> np.ndarray
    
    @property
    def time_window(self) -> List[float]
    
    @property
    def threat(self) -> float
    
    @property
    def execution_time(self) -> float
```

##### TaskManager
```python
class TaskManager(EntityManager):
    """Manager for task collections."""
    
    def __init__(self, task_list: List[Task] = None)
    
    def get_unassigned_tasks(self) -> List[Task]
    
    def get_tasks_in_time_window(self, current_time: float) -> List[Task]
    
    def calculate_total_required_resources(self) -> np.ndarray
    
    def get_high_priority_tasks(self, threshold: float = 0.7) -> List[Task]
    
    def get_tasks_by_threat_level(self, min_threat: float, max_threat: float) -> List[Task]
    
    def update_task_states(self, assignments: Dict[int, List[int]]) -> None
```

##### TaskGenParams
```python
@dataclass
class TaskGenParams(GenParams):
    """Parameters for task generation."""
    
    required_resources_range: Tuple[float, float] = (15.0, 35.0)
    time_window_ranges: List[Tuple[float, float]] = field(default_factory=list)
    threat_range: Tuple[float, float] = (0.1, 0.8)
    execution_time_range: Tuple[float, float] = (1.0, 5.0)
```

#### Enums

##### TaskState
```python
class TaskState(Enum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"
```

#### Functions

```python
def generate_task_dict_list(
    task_num: int,
    params: TaskGenParams,
    seed: int = None
) -> List[Dict]
```

### framework.coalition_manager

#### Classes

##### CoalitionManager
```python
class CoalitionManager:
    """Manages UAV coalitions and task assignments."""
    
    def __init__(self, uav_ids: List[int], task_ids: List[int])
    
    def assign_coalition_to_task(self, task_id: int, uav_coalition: List[int]) -> None
    
    def get_task_coalition(self, task_id: int) -> List[int]
    
    def get_uav_assignment(self, uav_id: int) -> int | None
    
    def is_uav_available(self, uav_id: int) -> bool
    
    def remove_assignment(self, task_id: int) -> None
    
    def clear_all_assignments(self) -> None
    
    def get_completion_rate(self) -> float
    
    def get_assignment_summary(self) -> Dict
    
    def get_unassigned_tasks(self) -> List[int]
    
    def get_unassigned_uavs(self) -> List[int]
    
    def validate_assignments(self, uav_manager: UAVManager, task_manager: TaskManager) -> bool
    
    def calculate_total_utility(self, uav_manager: UAVManager, task_manager: TaskManager) -> float
```

### framework.mrta_solver

#### Classes

##### MRTASolver
```python
class MRTASolver:
    """Base class for Multi-Robot Task Assignment solvers."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    @staticmethod
    def uav_type() -> Type[UAV]
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    # Properties
    @property
    def uav_manager(self) -> UAVManager
    
    @property
    def task_manager(self) -> TaskManager
    
    @property
    def coalition_manager(self) -> CoalitionManager
    
    @property
    def hyper_params(self) -> HyperParams
```

### framework.dynamic_mrta_solver

#### Classes

##### DynamicEnvironmentConfig
```python
@dataclass
class DynamicEnvironmentConfig:
    """Configuration for dynamic environment simulation."""
    
    new_task_rate: float = 0.1
    task_completion_time: float = 5.0
    max_simulation_time: float = 100.0
    uav_failure_rate: float = 0.0
    communication_range: float = 500.0
```

##### DynamicMRTASolver
```python
class DynamicMRTASolver(MRTASolver):
    """Solver for dynamic task assignment scenarios."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        dynamic_config: DynamicEnvironmentConfig
    )
    
    def run_dynamic_simulation(
        self,
        sim_env: SimulationEnv,
        show_progress: bool = False
    ) -> Dict
    
    def handle_new_task(self, task: Task) -> None
    
    def handle_task_completion(self, task_id: int) -> None
    
    def handle_uav_failure(self, uav_id: int) -> None
    
    def rebalance_assignments(self) -> None
```

### framework.sim

#### Classes

##### SimulationEnv
```python
class SimulationEnv:
    """Simulation environment for UAV task assignment."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        time_step: float = 0.1
    )
    
    def step(self) -> bool
    
    def run_simulation(
        self,
        max_time: float = 100.0,
        show_animation: bool = False
    ) -> Dict
    
    def get_state(self) -> SimState
    
    def reset(self) -> None
    
    def add_task_during_simulation(self, task: Task) -> None
    
    def remove_uav_during_simulation(self, uav_id: int) -> None
    
    # Properties
    @property
    def current_time(self) -> float
    
    @property
    def time_step(self) -> float
```

#### Enums

##### SimState
```python
class SimState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
```

### framework.utils

#### Classes

##### EvalResult
```python
@dataclass
class EvalResult:
    """Evaluation results for solver performance."""
    
    completion_rate: float = 0.0
    resource_use_rate: float = 0.0
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_exploss: float = 0.0
    elapsed_time: float = 0.0
    
    def format_print(self) -> None
    
    def to_dict(self) -> Dict
    
    def to_flattened_dict(self) -> Dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> EvalResult
```

#### Functions

```python
def get_resources_weights(
    required_resources: np.ndarray,
    task_obtained_resources: np.ndarray
) -> np.ndarray

def calculate_map_shape_on_mana(
    uav_manager: UAVManager,
    task_manager: TaskManager
) -> List[float]

def calculate_map_shape_on_list(
    uav_list: List[UAV],
    task_list: List[Task]
) -> List[float]

def calculate_map_shape_on_dict_list(
    uav_dict_list: List[Dict],
    task_dict_list: List[Dict]
) -> List[float]

def calculate_obtained_resources(
    coalition: List[int],
    uav_manager: UAVManager,
    resources_num: int
) -> np.ndarray

def calculate_obtained_resources_path_cost_weight(
    uav_coalition: List[UAV],
    resources_num: int
) -> Tuple[np.ndarray, float, float]

def format_with_prettier(json_string: str) -> str
```

### framework.test

#### Classes

##### SaveResult
```python
@dataclass
class SaveResult:
    """Data structure for saving experimental results."""
    
    solver_name: str
    test_case: str
    eval_results: List[EvalResult]
    parameters: Dict
    timestamp: str
    
    def save_to_file(self, filepath: str) -> None
    
    @classmethod
    def load_from_file(cls, filepath: str) -> SaveResult
```

##### TestFramework
```python
class TestFramework:
    """Base class for running algorithm tests."""
    
    def run_test_with_simulation(
        self,
        solver_class: Type[MRTASolver],
        test_case_path: str,
        show_animation: bool = False
    ) -> Dict
    
    def evaluate_solver_performance(
        self,
        solver: MRTASolver,
        start_time: float,
        end_time: float
    ) -> EvalResult
```

##### TestNums
```python
class TestNums(TestFramework):
    """Test framework for scalability experiments."""
    
    @staticmethod
    def run_vary_uav_nums(
        uav_nums: List[int],
        solver_types: List[Type[MRTASolver]],
        task_num: int = 20,
        test_times: int = 5
    ) -> Dict
    
    @staticmethod
    def run_vary_task_nums(
        task_nums: List[int],
        solver_types: List[Type[MRTASolver]],
        uav_num: int = 40,
        test_times: int = 5
    ) -> Dict
```

##### TestHyperParams
```python
class TestHyperParams(TestFramework):
    """Test framework for hyperparameter sensitivity analysis."""
    
    @staticmethod
    def run_vary_hyper_params(
        param_name: str,
        param_values: List[float],
        solver_types: List[Type[MRTASolver]],
        uav_num: int = 40,
        task_num: int = 20,
        test_times: int = 5
    ) -> Dict
```

#### Functions

```python
def run_on_test_case(
    test_case_path: str,
    solver_types: List[Type[MRTASolver]],
    test_times: int = 1,
    timeout: int = 60,
    show_simulation: bool = False
) -> Dict
```

---

## Solvers Module

### solvers.csci2024

#### Classes

##### ChinaScience2024_CoalitionFormationGame
```python
class ChinaScience2024_CoalitionFormationGame(MRTASolver):
    """Coalition Formation Game solver from CSCI 2024 paper."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def calculate_uav_utility(
        self,
        uav_id: int,
        task_id: int,
        coalition: List[int]
    ) -> float
    
    def calculate_coalition_utility(
        self,
        task_id: int,
        coalition: List[int]
    ) -> float
```

### solvers.iros2024

#### Classes

##### IROS2024_CoalitionFormationGame
```python
class IROS2024_CoalitionFormationGame(MRTASolver):
    """Coalition Formation Game solver from IROS 2024 paper."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def calculate_switching_utilities(
        self,
        uav_id: int,
        current_task: int,
        target_task: int
    ) -> Tuple[float, float, float, float]
```

##### IROS2024_CoalitionFormationGame_2
```python
class IROS2024_CoalitionFormationGame_2(IROS2024_CoalitionFormationGame):
    """Enhanced version of IROS2024 solver."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
```

### solvers.icra2024

#### Classes

##### Message
```python
@dataclass
class Message:
    """Communication message between UAVs."""
    
    sender_id: int
    receiver_id: int
    message_type: str
    content: Dict
    timestamp: float
```

##### AutoUAV
```python
class AutoUAV(UAV):
    """Autonomous UAV with communication capabilities."""
    
    def __init__(self, *args, **kwargs)
    
    def send_message(self, message: Message) -> None
    
    def receive_message(self, message: Message) -> None
    
    def process_messages(self) -> None
    
    def make_local_decision(self) -> Dict
```

##### ICRA2024_CoalitionFormationGame
```python
class ICRA2024_CoalitionFormationGame(MRTASolver):
    """Coalition Formation Game solver from ICRA 2024 paper."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def setup_communication_network(self) -> None
    
    def simulate_message_passing(self) -> None
```

### solvers.auction_solver

#### Classes

##### Bid
```python
@dataclass
class Bid:
    """Auction bid representation."""
    
    bidder_id: int
    task_id: int
    bid_value: float
    coalition_members: List[int]
    timestamp: float
```

##### Coalition
```python
@dataclass
class Coalition:
    """Coalition representation for auctions."""
    
    members: List[int]
    task_id: int
    total_value: float
    last_updated: float
```

##### AuctionBiddingSolver
```python
class AuctionBiddingSolver(MRTASolver):
    """Auction-based task assignment solver."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def conduct_auction_round(self) -> bool
    
    def collect_bids(self, task_id: int) -> List[Bid]
    
    def determine_winners(self, bids: List[Bid]) -> List[Bid]
    
    def calculate_bid_value(
        self,
        uav_id: int,
        task_id: int,
        coalition: List[int]
    ) -> float
    
    # Properties
    @property
    def max_coalition_size(self) -> int
    
    @property
    def bid_decay_factor(self) -> float
    
    @property
    def min_bid_improvement(self) -> float
```

##### AuctionBiddingSolverAdvanced
```python
class AuctionBiddingSolverAdvanced(AuctionBiddingSolver):
    """Advanced auction solver with sophisticated strategies."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def adaptive_bidding_strategy(self, uav_id: int, task_id: int) -> float
    
    def coalition_stability_analysis(self, coalition: List[int], task_id: int) -> float
```

### solvers.distributed_solver

#### Classes

##### Message
```python
@dataclass
class Message:
    """Distributed communication message."""
    
    sender_id: int
    receiver_id: int
    message_type: str
    content: Dict
    timestamp: float
```

##### AutoUAV
```python
class AutoUAV(UAV):
    """Autonomous UAV for distributed algorithms."""
    
    def __init__(self, *args, **kwargs)
    
    def local_task_assignment(self) -> int | None
    
    def negotiate_with_neighbors(self) -> Dict
    
    def update_local_state(self, global_info: Dict) -> None
```

##### DistributedSolver
```python
class DistributedSolver(MRTASolver):
    """Distributed multi-agent task assignment solver."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def setup_communication_topology(self) -> None
    
    def run_consensus_algorithm(self) -> None
    
    def aggregate_local_decisions(self) -> Dict
```

### solvers.milp_solver

#### Classes

##### MILPSolver
```python
class MILPSolver(MRTASolver):
    """Mixed-Integer Linear Programming solver."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def formulate_milp_problem(self) -> Any  # PuLP problem object
    
    def extract_solution(self, problem: Any) -> Dict
    
    def validate_solution(self, solution: Dict) -> bool
```

##### MILPSolverPyomo
```python
class MILPSolverPyomo(MRTASolver):
    """MILP solver using Pyomo framework."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
```

### solvers.nlp_solver

#### Classes

##### NLPSolverScipy
```python
class NLPSolverScipy(MRTASolver):
    """Non-Linear Programming solver using SciPy."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def objective_function(self, x: np.ndarray) -> float
    
    def constraint_functions(self, x: np.ndarray) -> List[float]
    
    def setup_optimization_problem(self) -> Dict
```

##### NLPSolverPyomo
```python
class NLPSolverPyomo(MRTASolver):
    """NLP solver using Pyomo framework."""
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
```

### solvers.enum_solver

#### Classes

##### EnumerationSolver
```python
class EnumerationSolver(MRTASolver):
    """Exhaustive enumeration solver for small problems."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def enumerate_all_assignments(self) -> List[Dict]
    
    def evaluate_assignment(self, assignment: Dict) -> float
    
    def prune_infeasible_assignments(self, assignments: List[Dict]) -> List[Dict]
```

### solvers.brute_force_solver

#### Classes

##### SearchState
```python
@dataclass
class SearchState:
    """State representation for brute force search."""
    
    assignments: Dict[int, List[int]]
    utility: float
    depth: int
```

##### BruteForceSearchSolver
```python
class BruteForceSearchSolver(MRTASolver):
    """Brute force search solver with pruning."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams
    )
    
    @classmethod
    def type_name(cls) -> str
    
    def run_allocate(self, init_method: str = "random") -> CoalitionManager
    
    def depth_first_search(self, state: SearchState) -> SearchState
    
    def generate_successor_states(self, state: SearchState) -> List[SearchState]
    
    def is_goal_state(self, state: SearchState) -> bool
    
    def prune_state(self, state: SearchState) -> bool
```

### solvers.utils

#### Classes

##### Activations
```python
class Activations:
    """Activation functions for neural network components."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray
```

##### MRTA_CFG_Model_HyperParams
```python
@dataclass
class MRTA_CFG_Model_HyperParams:
    """Hyperparameters for CFG model."""
    
    learning_rate: float = 0.001
    batch_size: int = 32
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.1
    regularization: float = 0.001
```

##### MRTA_CFG_Model
```python
class MRTA_CFG_Model:
    """Neural network model for Coalition Formation Game."""
    
    def __init__(self, hyperparams: MRTA_CFG_Model_HyperParams)
    
    def forward(self, x: np.ndarray) -> np.ndarray
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None
    
    def predict(self, X: np.ndarray) -> np.ndarray
    
    def save_model(self, filepath: str) -> None
    
    def load_model(self, filepath: str) -> None
```

#### Functions

```python
def calculate_path_cost(
    uav_position: Point,
    task_position: Point,
    max_speed: float
) -> float

def calculate_threat_loss(
    uav_value: float,
    task_threat: float
) -> float

def normalize_features(
    features: np.ndarray,
    method: str = "minmax"
) -> np.ndarray
```

### solvers.driver

#### Functions

```python
def get_SolverTypes(choices: List[str]) -> List[Type[MRTASolver]]

def get_solver_by_name(name: str) -> Type[MRTASolver]

def list_available_solvers() -> List[str]
```

---

## Main Scripts

### main.py

#### Classes

##### CmdArgs
```python
@dataclass
class CmdArgs:
    output_path: Path
```

##### TestCmdArgs
```python
@dataclass
class TestCmdArgs(CmdArgs):
    test_case: str
    choices: List[str]
    timeout: float
    uav_nums: List[int]
    task_nums: List[int]
    random_test_times: int
    hp_values: List
    show: bool
    save_dir: Path
    sim: bool
```

##### PlotCmdArgs
```python
@dataclass
class PlotCmdArgs(CmdArgs):
    file_path: str
    x: str
    labels: List[str]
    choices: List[str]
    save_dir: Path
    show: bool
```

##### DynamicCmdArgs
```python
@dataclass
class DynamicCmdArgs(CmdArgs):
    test_case: str
    uav_nums: List[int]
    task_nums: List[int]
```

#### Functions

```python
def init_cmd_args() -> argparse.Namespace

def run_test_driver(cmd_args: TestCmdArgs) -> None

def run_show_driver(cmd_args: PlotCmdArgs) -> None

def run_dynamic_driver(cmd_args: DynamicCmdArgs) -> None

def main() -> None
```

### gen.py

#### Functions

```python
def gen_iros2024_case(out_path: str) -> None

def gen_icra2024_case(out_path: str) -> None

def gen_csci2024_data(out_path: str) -> None

def generate_random_scenario(
    uav_num: int,
    task_num: int,
    scenario_type: str = "balanced"
) -> Dict
```

---

## Constants and Enums

### Global Constants

```python
DEFAULT_EXECUTION_TIME = 2
DEFAULT_MAX_ITER = 100
DEFAULT_TOLERANCE = 1e-6
DEFAULT_RANDOM_SEED = 42

# Evaluation labels
ALL_LABELS = [
    "elapsed_time",
    "completion_rate", 
    "resource_use_rate",
    "total_distance",
    "total_energy",
    "total_exploss"
]
```

### Solver Names

```python
SOLVER_NAMES = {
    "csci2024": "ChinaScience2024_CoalitionFormationGame",
    "iros2024": "IROS2024_CoalitionFormationGame", 
    "icra2024": "ICRA2024_CoalitionFormationGame",
    "auction": "AuctionBiddingSolver",
    "distributed": "DistributedSolver",
    "milp": "MILPSolver",
    "nlp": "NLPSolverScipy",
    "enum": "EnumerationSolver",
    "brute_force": "BruteForceSearchSolver"
}
```

---

## Type Definitions

### Common Type Aliases

```python
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, Type

# Position types
Position = Union[List[float], Tuple[float, float, float], np.ndarray, Point]

# Resource types
ResourceVector = Union[List[float], np.ndarray]

# Time types
TimeWindow = Tuple[float, float]

# Assignment types
Assignment = Dict[int, List[int]]  # task_id -> uav_ids
UAVAssignment = Dict[int, int]     # uav_id -> task_id

# Evaluation types
MetricValue = Union[int, float]
EvaluationMetrics = Dict[str, MetricValue]

# Solver types
SolverClass = Type[MRTASolver]
SolverResult = CoalitionManager

# Test types
TestResult = Dict[str, Any]
TestResults = Dict[str, Dict[str, List[EvalResult]]]

# Configuration types
HyperParameterValue = Union[int, float, str, bool]
Configuration = Dict[str, HyperParameterValue]
```

### Function Signatures

```python
# Utility function signatures
UtilityFunction = Callable[[int, int, List[int]], float]  # uav_id, task_id, coalition -> utility
CostFunction = Callable[[Point, Point], float]           # start, end -> cost
EvaluationFunction = Callable[[CoalitionManager, UAVManager, TaskManager], EvalResult]

# Solver factory
SolverFactory = Callable[[UAVManager, TaskManager, CoalitionManager, HyperParams], MRTASolver]

# Test function signatures
TestFunction = Callable[[List[int], List[SolverClass]], TestResults]
```

### Exception Types

```python
class MRTAError(Exception):
    """Base exception for MRTA framework."""
    pass

class SolverError(MRTAError):
    """Exception raised by solvers."""
    pass

class ConfigurationError(MRTAError):
    """Exception for configuration errors."""
    pass

class ValidationError(MRTAError):
    """Exception for validation errors."""
    pass

class TimeoutError(MRTAError):
    """Exception for timeout errors."""
    pass
```

This comprehensive API reference provides complete documentation for all public interfaces in the Swarm UAV Task Assignment framework, enabling developers to effectively use and extend the system.