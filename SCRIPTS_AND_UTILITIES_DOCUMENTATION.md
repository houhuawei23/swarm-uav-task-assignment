# Scripts and Utilities Documentation

## Overview

This document provides comprehensive documentation for the main scripts, utilities, and helper functions in the Swarm UAV Task Assignment framework.

## Table of Contents

- [Main Entry Points](#main-entry-points)
- [Test Case Generation](#test-case-generation)
- [Utility Functions](#utility-functions)
- [Testing Framework](#testing-framework)
- [Simulation Environment](#simulation-environment)
- [Data Analysis and Plotting](#data-analysis-and-plotting)

---

## Main Entry Points

### main.py

The primary entry point for running experiments, tests, and visualizations.

#### Command Structure
```bash
python main.py <command> [options]
```

#### Available Commands

##### 1. Test Command
Runs solver algorithms with various configurations and parameters.

```bash
python main.py test [options]
```

**Options:**
- `--test_case`: Test case type or file path
  - `"uav_num"`: Vary number of UAVs
  - `"task_num"`: Vary number of tasks  
  - `"hyper_params.xxx"`: Vary hyperparameter xxx
  - `<file_path>`: Path to specific test case JSON file

- `--uav_nums`: List of UAV numbers to test (default: [40])
- `--task_nums`: List of task numbers to test (default: [20])
- `--choices`: Solver algorithms to test
- `--timeout`: Timeout for each algorithm in seconds (default: 10)
- `--random_test_times`: Number of random test iterations (default: 5)
- `--show`: Show plots during execution
- `--save_dir`: Directory to save results
- `--sim`: Enable simulation mode with time steps
- `-o, --output`: Output file path

**Usage Examples:**

```bash
# Test UAV number scalability
python main.py test --test_case uav_num --uav_nums 10 20 30 40 50 \
    --task_nums 10 --choices csci2024 iros2024 icra2024 \
    --random_test_times 3 --save_dir results/scalability

# Test task number scalability  
python main.py test --test_case task_num --uav_nums 40 \
    --task_nums 5 10 15 20 25 --choices auction milp \
    --timeout 60 --show

# Test hyperparameter sensitivity
python main.py test --test_case hyper_params.alpha \
    --hp_values 0.1 0.3 0.5 0.7 0.9 --choices csci2024 \
    --random_test_times 5

# Test specific test case file
python main.py test --test_case test_cases/custom_scenario.json \
    --choices csci2024 iros2024 auction --show --sim
```

**Available Solver Choices:**
- `csci2024`: ChinaScience2024_CoalitionFormationGame
- `iros2024`: IROS2024_CoalitionFormationGame
- `icra2024`: ICRA2024_CoalitionFormationGame
- `auction`: AuctionBiddingSolver
- `distributed`: DistributedSolver
- `milp`: MILPSolver
- `nlp`: NLPSolverScipy
- `enum`: EnumerationSolver
- `brute_force`: BruteForceSearchSolver

##### 2. Plot Command
Visualizes results from previous test runs.

```bash
python main.py plot [options]
```

**Options:**
- `-f, --file_path`: Path to results JSON file
- `-x, --xlabel`: X-axis variable (default: "uav_num")
- `--labels`: Metrics to plot (default: ["elapsed_time"])
- `--choices`: Solver algorithms to include in plot
- `--save_dir`: Directory to save plot images
- `--show`: Display plots interactively

**Available Labels:**
- `elapsed_time`: Algorithm execution time
- `completion_rate`: Task completion percentage
- `resource_use_rate`: Resource utilization percentage
- `total_distance`: Total travel distance
- `total_energy`: Total energy consumption
- `total_exploss`: Total expected loss

**Usage Examples:**

```bash
# Plot completion rate vs UAV number
python main.py plot -f results/scalability_test.json \
    -x uav_num --labels completion_rate resource_use_rate \
    --choices csci2024 iros2024 --show

# Plot multiple metrics
python main.py plot -f results/comparison.json \
    --labels elapsed_time completion_rate total_energy \
    --save_dir plots/comparison

# Plot hyperparameter sensitivity
python main.py plot -f results/hyperparam_test.json \
    -x alpha --labels completion_rate --show
```

##### 3. Dynamic Command
Runs dynamic simulation scenarios.

```bash
python main.py dynamic [options]
```

**Options:**
- `--test_case`: Test case file or scenario type
- `--uav_nums`: List of UAV numbers for dynamic scenarios
- `--task_nums`: List of task numbers for dynamic scenarios

**Usage Examples:**

```bash
# Dynamic simulation with varying parameters
python main.py dynamic --test_case dynamic_scenario \
    --uav_nums 20 30 --task_nums 10 15

# Dynamic simulation from file
python main.py dynamic --test_case test_cases/dynamic_test.json
```

---

## Test Case Generation

### gen.py

Generates test cases for different experimental scenarios.

#### Command Structure
```bash
python gen.py [options]
```

**Options:**
- `--output`: Output file path (default: "./tests/case1.json")
- `--uav_num`: Number of UAVs (default: 50)
- `--num_tasks`: Number of tasks (default: 5)
- `--choice`: Data generation scenario ("csci2024" or "iros2024")

#### Usage Examples

```bash
# Generate CSCI2024 scenario
python gen.py --output test_cases/csci_scenario.json \
    --uav_num 50 --num_tasks 10 --choice csci2024

# Generate IROS2024 scenario
python gen.py --output test_cases/iros_scenario.json \
    --uav_num 30 --num_tasks 8 --choice iros2024
```

#### Programmatic Usage

```python
from gen import gen_iros2024_case, gen_icra2024_case, gen_csci2024_data

# Generate IROS2024 test case
gen_iros2024_case("test_cases/iros_case.json")

# Generate ICRA2024 test case  
gen_icra2024_case("test_cases/icra_case.json")

# Generate CSCI2024 data
gen_csci2024_data("test_cases/csci_case.json")
```

#### Scenario Configurations

##### IROS2024 Scenario
```python
# UAV Parameters
uav_num = 50
resources_num = 5
region_ranges = [(100, 900), (100, 900), (0, 0)]  # 1000x1000m area
resources_range = (5, 10)
value_range = (25, 50)
speed_range = (20, 30)  # m/s
uav_mass_range = (10, 15)  # kg

# Task Parameters
num_tasks = 5
required_resources_range = (25, 45)
time_window_ranges = [(0, 10), (90, 100)]
threat_range = (0.1, 0.9)
```

##### ICRA2024 Scenario
```python
# Similar to IROS2024 but with communication parameters
comm_bandwidth_range = (5, 15)  # kHz
trans_power_range = (20, 30)  # dBm
```

---

## Utility Functions

### framework.utils

#### Evaluation Functions

```python
from framework.utils import EvalResult, calculate_obtained_resources

@dataclass
class EvalResult:
    """Comprehensive evaluation metrics for solver performance."""
    completion_rate: float = 0.0      # Percentage of tasks completed
    resource_use_rate: float = 0.0    # Percentage of resources utilized
    total_distance: float = 0.0       # Total travel distance
    total_energy: float = 0.0         # Total energy consumption
    total_exploss: float = 0.0        # Total expected loss
    elapsed_time: float = 0.0         # Algorithm execution time
    
    def format_print(self):
        """Print formatted evaluation results."""
        print(f" Completion Rate: {self.completion_rate:.2f}")
        print(f" Resource Use Rate: {self.resource_use_rate:.2f}")
        print(f" Total Distance: {self.total_distance:.2f}")
        print(f" Total Energy: {self.total_energy:.2f}")
        print(f" Total Exploss: {self.total_exploss:.2f}")
        print(f" Elapsed Time: {self.elapsed_time:.2f}")
```

**Usage Example:**
```python
# Evaluate solver performance
eval_result = EvalResult(
    completion_rate=0.85,
    resource_use_rate=0.72,
    total_distance=1250.5,
    total_energy=890.2,
    elapsed_time=2.34
)

eval_result.format_print()

# Convert to dictionary for saving
result_dict = eval_result.to_dict()
```

#### Resource Calculation Functions

```python
def calculate_obtained_resources(
    coalition: List[int], 
    uav_manager: UAVManager, 
    resources_num: int
) -> np.ndarray:
    """Calculate total resources available from UAV coalition.
    
    Args:
        coalition: List of UAV IDs in the coalition
        uav_manager: UAV manager instance
        resources_num: Number of resource types
        
    Returns:
        Array of total resources available
    """

def get_resources_weights(
    required_resources: np.ndarray, 
    task_obtained_resources: np.ndarray
) -> np.ndarray:
    """Calculate resource weights based on remaining requirements.
    
    Args:
        required_resources: Resources required by task
        task_obtained_resources: Resources already allocated
        
    Returns:
        Normalized weights for remaining resources
    """
```

**Usage Example:**
```python
# Calculate coalition resources
coalition_ids = [1, 2, 3]
total_resources = calculate_obtained_resources(
    coalition_ids, uav_manager, resources_num=5
)

# Calculate resource priorities
required = np.array([20, 15, 10, 8, 12])
available = np.array([15, 10, 5, 8, 8])
weights = get_resources_weights(required, available)
```

#### Map and Spatial Functions

```python
def calculate_map_shape_on_mana(
    uav_manager: UAVManager, 
    task_manager: TaskManager
) -> List[float]:
    """Calculate map boundaries from entity positions.
    
    Returns:
        [max_x + 1, max_y + 1, max_z + 1]
    """

def calculate_map_shape_on_dict_list(
    uav_dict_list: List[Dict], 
    task_dict_list: List[Dict]
) -> List[float]:
    """Calculate map boundaries from dictionary data."""
```

#### Formatting Functions

```python
def format_with_prettier(json_string: str) -> str:
    """Format JSON string using Prettier for better readability.
    
    Args:
        json_string: Raw JSON string
        
    Returns:
        Formatted JSON string
    """
```

**Usage Example:**
```python
import json
from framework.utils import format_with_prettier

# Format JSON output
data = {"uavs": [...], "tasks": [...]}
json_str = json.dumps(data)
formatted_json = format_with_prettier(json_str)

with open("formatted_output.json", "w") as f:
    f.write(formatted_json)
```

---

## Testing Framework

### framework.test

#### TestFramework Base Class

```python
from framework.test import TestFramework

class TestFramework:
    """Base class for running algorithm tests and experiments."""
    
    def run_test_with_simulation(
        self,
        solver_class: Type[MRTASolver],
        test_case_path: str,
        show_animation: bool = False
    ) -> Dict:
        """Run test with optional simulation visualization."""
```

#### TestNums - Scalability Testing

```python
from framework.test import TestNums

class TestNums(TestFramework):
    """Test framework for scalability experiments."""
    
    @staticmethod
    def run_vary_uav_nums(
        uav_nums: List[int],
        solver_types: List[Type[MRTASolver]],
        task_num: int = 20,
        test_times: int = 5
    ) -> Dict:
        """Test solver performance with varying UAV numbers."""
        
    @staticmethod
    def run_vary_task_nums(
        task_nums: List[int],
        solver_types: List[Type[MRTASolver]],
        uav_num: int = 40,
        test_times: int = 5
    ) -> Dict:
        """Test solver performance with varying task numbers."""
```

**Usage Example:**
```python
from solvers import ChinaScience2024_CoalitionFormationGame, IROS2024_CoalitionFormationGame

# Test UAV scalability
solver_types = [
    ChinaScience2024_CoalitionFormationGame,
    IROS2024_CoalitionFormationGame
]

results = TestNums.run_vary_uav_nums(
    uav_nums=[10, 20, 30, 40, 50],
    solver_types=solver_types,
    task_num=15,
    test_times=3
)

# Results structure: {solver_name: {uav_num: [EvalResult, ...]}}
for solver_name, solver_results in results.items():
    print(f"Results for {solver_name}:")
    for uav_num, eval_results in solver_results.items():
        avg_completion = np.mean([r.completion_rate for r in eval_results])
        print(f"  {uav_num} UAVs: {avg_completion:.2f} completion rate")
```

#### TestHyperParams - Parameter Sensitivity

```python
from framework.test import TestHyperParams

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
    ) -> Dict:
        """Test solver sensitivity to hyperparameter changes."""
```

**Usage Example:**
```python
# Test alpha parameter sensitivity
results = TestHyperParams.run_vary_hyper_params(
    param_name="hyper_params.alpha",
    param_values=[0.1, 0.3, 0.5, 0.7, 0.9],
    solver_types=[ChinaScience2024_CoalitionFormationGame],
    test_times=5
)
```

#### SaveResult - Results Management

```python
from framework.test import SaveResult

@dataclass
class SaveResult:
    """Data structure for saving experimental results."""
    
    solver_name: str
    test_case: str
    eval_results: List[EvalResult]
    parameters: Dict
    timestamp: str
    
    def save_to_file(self, filepath: str):
        """Save results to JSON file."""
        
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load results from JSON file."""
```

---

## Simulation Environment

### framework.sim

#### SimulationEnv

```python
from framework.sim import SimulationEnv, SimState

class SimState(Enum):
    """Simulation states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"

class SimulationEnv:
    """Simulation environment for UAV task assignment."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        time_step: float = 0.1
    ):
        """Initialize simulation environment."""
        
    def step(self) -> bool:
        """Execute one simulation step."""
        
    def run_simulation(
        self,
        max_time: float = 100.0,
        show_animation: bool = False
    ) -> Dict:
        """Run complete simulation."""
        
    def get_state(self) -> SimState:
        """Get current simulation state."""
        
    def reset(self):
        """Reset simulation to initial state."""
```

**Usage Example:**
```python
# Create simulation environment
sim_env = SimulationEnv(
    uav_manager=uav_manager,
    task_manager=task_manager,
    time_step=0.1
)

# Run simulation with visualization
results = sim_env.run_simulation(
    max_time=50.0,
    show_animation=True
)

print(f"Simulation completed in {results['total_time']} seconds")
print(f"Tasks completed: {results['completed_tasks']}")
```

---

## Data Analysis and Plotting

### Plotting Utilities

The framework includes comprehensive plotting capabilities for result visualization.

#### Available Plot Types

1. **Scalability Plots**: Performance vs. problem size
2. **Comparison Plots**: Multiple algorithms comparison
3. **Time Series**: Performance over time
4. **Box Plots**: Statistical distribution of results
5. **Heatmaps**: Parameter sensitivity analysis

#### Usage Examples

```python
# Generate scalability comparison plot
python main.py plot -f results/scalability.json \
    -x uav_num --labels completion_rate elapsed_time \
    --choices csci2024 iros2024 auction \
    --save_dir plots/scalability --show

# Generate hyperparameter sensitivity heatmap
python main.py plot -f results/hyperparam.json \
    -x alpha --labels completion_rate \
    --save_dir plots/sensitivity
```

#### Custom Plotting

```python
import matplotlib.pyplot as plt
import json
from framework.utils import EvalResult

# Load and process results
with open("results/experiment.json", "r") as f:
    data = json.load(f)

# Create custom plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot completion rate
for solver_name, results in data.items():
    uav_nums = list(results.keys())
    completion_rates = [np.mean([r['completion_rate'] for r in results[uav_num]]) 
                       for uav_num in uav_nums]
    axes[0,0].plot(uav_nums, completion_rates, label=solver_name, marker='o')

axes[0,0].set_xlabel('Number of UAVs')
axes[0,0].set_ylabel('Completion Rate')
axes[0,0].legend()
axes[0,0].grid(True)

plt.tight_layout()
plt.savefig('custom_analysis.png', dpi=300)
plt.show()
```

---

## Batch Processing and Automation

### Batch Experiment Scripts

```bash
#!/bin/bash
# batch_experiments.sh - Run comprehensive experiments

# Scalability tests
python main.py test --test_case uav_num --uav_nums 10 20 30 40 50 \
    --task_nums 10 --choices csci2024 iros2024 icra2024 auction \
    --random_test_times 5 --save_dir results/scalability_uav

python main.py test --test_case task_num --uav_nums 40 \
    --task_nums 5 10 15 20 25 --choices csci2024 iros2024 \
    --random_test_times 5 --save_dir results/scalability_task

# Hyperparameter sensitivity
for param in alpha beta gamma; do
    python main.py test --test_case hyper_params.$param \
        --hp_values 0.1 0.3 0.5 0.7 0.9 --choices csci2024 \
        --random_test_times 3 --save_dir results/hyperparam_$param
done

# Generate plots
python main.py plot -f results/scalability_uav/results.json \
    -x uav_num --labels completion_rate elapsed_time \
    --save_dir plots/scalability --show
```

### Python Automation

```python
import subprocess
import json
from pathlib import Path

def run_comprehensive_experiments():
    """Run a comprehensive set of experiments."""
    
    # Define experiment configurations
    experiments = [
        {
            "name": "uav_scalability",
            "test_case": "uav_num",
            "uav_nums": [10, 20, 30, 40, 50],
            "task_nums": [10],
            "choices": ["csci2024", "iros2024", "auction"],
            "test_times": 5
        },
        {
            "name": "task_scalability", 
            "test_case": "task_num",
            "uav_nums": [40],
            "task_nums": [5, 10, 15, 20, 25],
            "choices": ["csci2024", "iros2024"],
            "test_times": 5
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"Running experiment: {exp['name']}")
        
        cmd = [
            "python", "main.py", "test",
            "--test_case", exp["test_case"],
            "--uav_nums"] + [str(n) for n in exp["uav_nums"]] + [
            "--task_nums"] + [str(n) for n in exp["task_nums"]] + [
            "--choices"] + exp["choices"] + [
            "--random_test_times", str(exp["test_times"]),
            "--save_dir", f"results/{exp['name']}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {exp['name']} completed successfully")
            results[exp['name']] = "success"
        else:
            print(f"✗ {exp['name']} failed: {result.stderr}")
            results[exp['name']] = "failed"
    
    return results

# Run experiments
if __name__ == "__main__":
    results = run_comprehensive_experiments()
    print("Experiment Summary:", results)
```

This comprehensive documentation provides detailed information about all scripts, utilities, and helper functions in the framework, enabling users to effectively utilize all available tools for their MRTA research and applications.