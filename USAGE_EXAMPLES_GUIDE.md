# Usage Examples and Quick Start Guide

## Overview

This guide provides practical, step-by-step examples for using the Swarm UAV Task Assignment framework. From basic setup to advanced research scenarios, these examples will help you get started quickly and effectively.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Usage Examples](#basic-usage-examples)
- [Advanced Scenarios](#advanced-scenarios)
- [Research and Benchmarking](#research-and-benchmarking)
- [Custom Development](#custom-development)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd swarm-uav-task-assignment

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

### Your First Test Run

```bash
# Generate a test case
python gen.py --output test_cases/quickstart.json --uav_num 20 --num_tasks 8 --choice iros2024

# Run a simple comparison
python main.py test --test_case test_cases/quickstart.json \
    --choices csci2024 iros2024 --show

# View results
ls results/
```

---

## Basic Usage Examples

### Example 1: Simple Task Assignment

Create and run a basic UAV task assignment scenario.

```python
#!/usr/bin/env python3
"""
Example 1: Basic UAV Task Assignment
Creates a simple scenario with 5 UAVs and 3 tasks, then runs allocation.
"""

from framework import UAV, Task, UAVManager, TaskManager, CoalitionManager, HyperParams
from solvers import ChinaScience2024_CoalitionFormationGame
import numpy as np

def example_basic_assignment():
    # Create UAVs with different capabilities
    uavs = [
        UAV(id=1, position=[0, 0, 0], resources=[10, 8, 12], value=100, max_speed=25),
        UAV(id=2, position=[50, 0, 0], resources=[8, 10, 9], value=90, max_speed=30),
        UAV(id=3, position=[100, 0, 0], resources=[12, 6, 15], value=110, max_speed=20),
        UAV(id=4, position=[0, 50, 0], resources=[6, 12, 8], value=85, max_speed=28),
        UAV(id=5, position=[50, 50, 0], resources=[14, 9, 11], value=105, max_speed=22),
    ]
    
    # Create tasks with different requirements
    tasks = [
        Task(id=1, position=[200, 100, 0], required_resources=[15, 10, 8], 
             time_window=[0, 30], threat=0.2),
        Task(id=2, position=[150, 200, 0], required_resources=[12, 15, 12], 
             time_window=[10, 40], threat=0.4),
        Task(id=3, position=[300, 150, 0], required_resources=[20, 8, 18], 
             time_window=[5, 35], threat=0.1),
    ]
    
    # Create managers
    uav_manager = UAVManager(uavs)
    task_manager = TaskManager(tasks)
    coalition_manager = CoalitionManager(
        uav_ids=[uav.id for uav in uavs],
        task_ids=[task.id for task in tasks]
    )
    
    # Set hyperparameters
    hyper_params = HyperParams(max_iter=50, tolerance=1e-4)
    
    # Create and run solver
    solver = ChinaScience2024_CoalitionFormationGame(
        uav_manager, task_manager, coalition_manager, hyper_params
    )
    
    print("Running task assignment...")
    result_coalition_manager = solver.run_allocate()
    
    # Display results
    print("\n=== Assignment Results ===")
    completion_rate = result_coalition_manager.get_completion_rate()
    print(f"Task completion rate: {completion_rate:.2%}")
    
    for task_id in task_manager.get_ids():
        coalition = result_coalition_manager.get_task_coalition(task_id)
        task = task_manager.get(task_id)
        print(f"Task {task_id} (req: {task.required_resources}) -> UAVs: {coalition}")
        
        # Show resource satisfaction
        if coalition:
            total_resources = uav_manager.get_available_resources(coalition)
            satisfaction = np.minimum(total_resources / task.required_resources, 1.0)
            print(f"  Resource satisfaction: {satisfaction}")
    
    return result_coalition_manager

if __name__ == "__main__":
    example_basic_assignment()
```

### Example 2: Algorithm Comparison

Compare different solvers on the same problem.

```python
#!/usr/bin/env python3
"""
Example 2: Algorithm Comparison
Compares multiple solvers on the same problem instance.
"""

from framework import UAV, Task, UAVManager, TaskManager, CoalitionManager, HyperParams
from solvers import (
    ChinaScience2024_CoalitionFormationGame,
    IROS2024_CoalitionFormationGame,
    AuctionBiddingSolver,
    EnumerationSolver
)
from framework.utils import EvalResult
import time
import json

def example_algorithm_comparison():
    # Load test case from file
    with open("test_cases/quickstart.json", "r") as f:
        data = json.load(f)
    
    # Create entities from data
    uavs = [UAV.from_dict(uav_data) for uav_data in data["uavs"]]
    tasks = [Task.from_dict(task_data) for task_data in data["tasks"]]
    
    # Create managers
    uav_manager = UAVManager(uavs)
    task_manager = TaskManager(tasks)
    
    # Solvers to compare
    solvers_to_test = [
        ("CSCI2024", ChinaScience2024_CoalitionFormationGame),
        ("IROS2024", IROS2024_CoalitionFormationGame),
        ("Auction", AuctionBiddingSolver),
        ("Enumeration", EnumerationSolver),  # Only for small problems
    ]
    
    results = {}
    
    print("=== Algorithm Comparison ===")
    print(f"Problem: {len(uavs)} UAVs, {len(tasks)} tasks")
    
    for solver_name, solver_class in solvers_to_test:
        print(f"\nTesting {solver_name}...")
        
        # Create fresh coalition manager for each test
        coalition_manager = CoalitionManager(
            uav_ids=[uav.id for uav in uavs],
            task_ids=[task.id for task in tasks]
        )
        
        # Configure hyperparameters based on solver
        if solver_name == "Enumeration" and (len(uavs) > 8 or len(tasks) > 4):
            print(f"  Skipping {solver_name} - problem too large")
            continue
            
        hyper_params = HyperParams(max_iter=50 if solver_name != "Enumeration" else 1000)
        
        # Create solver
        solver = solver_class(uav_manager, task_manager, coalition_manager, hyper_params)
        
        # Run with timing
        start_time = time.time()
        try:
            result_coalition_manager = solver.run_allocate()
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            completion_rate = result_coalition_manager.get_completion_rate()
            
            # Store results
            results[solver_name] = {
                "completion_rate": completion_rate,
                "elapsed_time": elapsed_time,
                "assignments": result_coalition_manager.get_assignment_summary()
            }
            
            print(f"  ✓ Completion rate: {completion_rate:.2%}")
            print(f"  ✓ Time: {elapsed_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[solver_name] = {"error": str(e)}
    
    # Print comparison summary
    print("\n=== Comparison Summary ===")
    print(f"{'Solver':<15} {'Completion':<12} {'Time (s)':<10}")
    print("-" * 40)
    
    for solver_name, result in results.items():
        if "error" in result:
            print(f"{solver_name:<15} {'ERROR':<12} {'-':<10}")
        else:
            completion = f"{result['completion_rate']:.1%}"
            time_str = f"{result['elapsed_time']:.3f}"
            print(f"{solver_name:<15} {completion:<12} {time_str:<10}")
    
    # Save detailed results
    with open("results/comparison_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    example_algorithm_comparison()
```

### Example 3: Parameter Sensitivity Analysis

Analyze how hyperparameters affect solver performance.

```python
#!/usr/bin/env python3
"""
Example 3: Parameter Sensitivity Analysis
Tests how different parameter values affect algorithm performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from framework import HyperParams
from framework.test import TestHyperParams
from solvers import ChinaScience2024_CoalitionFormationGame

def example_parameter_sensitivity():
    # Test alpha parameter (resource weight)
    print("Testing alpha parameter sensitivity...")
    
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    solver_types = [ChinaScience2024_CoalitionFormationGame]
    
    results = TestHyperParams.run_vary_hyper_params(
        param_name="hyper_params.alpha",
        param_values=alpha_values,
        solver_types=solver_types,
        uav_num=30,
        task_num=10,
        test_times=3
    )
    
    # Extract and analyze results
    solver_name = "ChinaScience2024_CoalitionFormationGame"
    if solver_name in results:
        solver_results = results[solver_name]
        
        # Calculate statistics
        mean_completion = []
        std_completion = []
        mean_time = []
        
        for alpha in alpha_values:
            eval_results = solver_results[alpha]
            completion_rates = [r.completion_rate for r in eval_results]
            times = [r.elapsed_time for r in eval_results]
            
            mean_completion.append(np.mean(completion_rates))
            std_completion.append(np.std(completion_rates))
            mean_time.append(np.mean(times))
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Completion rate plot
        ax1.errorbar(alpha_values, mean_completion, yerr=std_completion, 
                    marker='o', capsize=5, capthick=2)
        ax1.set_xlabel('Alpha (Resource Weight)')
        ax1.set_ylabel('Completion Rate')
        ax1.set_title('Completion Rate vs Alpha Parameter')
        ax1.grid(True, alpha=0.3)
        
        # Execution time plot
        ax2.plot(alpha_values, mean_time, marker='s', color='red')
        ax2.set_xlabel('Alpha (Resource Weight)')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Execution Time vs Alpha Parameter')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_sensitivity_alpha.png', dpi=300)
        plt.show()
        
        # Print optimal parameter
        best_idx = np.argmax(mean_completion)
        optimal_alpha = alpha_values[best_idx]
        print(f"\nOptimal alpha value: {optimal_alpha}")
        print(f"Best completion rate: {mean_completion[best_idx]:.2%}")
        print(f"Execution time at optimal: {mean_time[best_idx]:.3f}s")
    
    return results

if __name__ == "__main__":
    example_parameter_sensitivity()
```

---

## Advanced Scenarios

### Example 4: Dynamic Task Assignment

Handle tasks that appear during execution.

```python
#!/usr/bin/env python3
"""
Example 4: Dynamic Task Assignment
Demonstrates handling of tasks that appear during mission execution.
"""

from framework import UAV, Task, UAVManager, TaskManager, CoalitionManager, HyperParams
from framework.dynamic_mrta_solver import DynamicMRTASolver, DynamicEnvironmentConfig
from framework.sim import SimulationEnv
import random
import time

def example_dynamic_assignment():
    # Create initial UAV fleet
    uavs = []
    for i in range(15):
        uav = UAV(
            id=i,
            position=[random.uniform(0, 100), random.uniform(0, 100), 0],
            resources=[random.uniform(5, 15) for _ in range(5)],
            value=random.uniform(80, 120),
            max_speed=random.uniform(20, 35)
        )
        uavs.append(uav)
    
    # Create initial tasks
    initial_tasks = []
    for i in range(5):
        task = Task(
            id=i,
            position=[random.uniform(200, 800), random.uniform(200, 800), 0],
            required_resources=[random.uniform(10, 30) for _ in range(5)],
            time_window=[0, 60],
            threat=random.uniform(0.1, 0.5)
        )
        initial_tasks.append(task)
    
    # Create managers
    uav_manager = UAVManager(uavs)
    task_manager = TaskManager(initial_tasks)
    coalition_manager = CoalitionManager(
        uav_ids=[uav.id for uav in uavs],
        task_ids=[task.id for task in initial_tasks]
    )
    
    # Configure dynamic environment
    dynamic_config = DynamicEnvironmentConfig(
        new_task_rate=0.1,  # Tasks per time step
        task_completion_time=5.0,
        max_simulation_time=100.0
    )
    
    # Create dynamic solver
    hyper_params = HyperParams(max_iter=20, tolerance=1e-3)
    dynamic_solver = DynamicMRTASolver(
        uav_manager, task_manager, coalition_manager, 
        hyper_params, dynamic_config
    )
    
    # Create simulation environment
    sim_env = SimulationEnv(uav_manager, task_manager, time_step=0.5)
    
    print("=== Dynamic Task Assignment Simulation ===")
    print(f"Initial: {len(uavs)} UAVs, {len(initial_tasks)} tasks")
    
    # Run dynamic simulation
    simulation_results = dynamic_solver.run_dynamic_simulation(
        sim_env, show_progress=True
    )
    
    # Display results
    print("\n=== Simulation Results ===")
    print(f"Total simulation time: {simulation_results['total_time']:.1f}s")
    print(f"Tasks completed: {simulation_results['completed_tasks']}")
    print(f"Tasks added during simulation: {simulation_results['new_tasks_added']}")
    print(f"Final completion rate: {simulation_results['final_completion_rate']:.2%}")
    print(f"Average response time: {simulation_results['avg_response_time']:.2f}s")
    
    return simulation_results

if __name__ == "__main__":
    example_dynamic_assignment()
```

### Example 5: Multi-Objective Optimization

Balance multiple objectives like completion time, energy consumption, and risk.

```python
#!/usr/bin/env python3
"""
Example 5: Multi-Objective Optimization
Demonstrates balancing multiple objectives in task assignment.
"""

from framework import UAV, Task, UAVManager, TaskManager, CoalitionManager, HyperParams
from solvers import ChinaScience2024_CoalitionFormationGame
from framework.utils import EvalResult
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class MultiObjectiveWeights:
    """Weights for different objectives."""
    completion_weight: float = 0.4
    energy_weight: float = 0.3
    risk_weight: float = 0.2
    time_weight: float = 0.1

def calculate_multi_objective_score(
    eval_result: EvalResult, 
    weights: MultiObjectiveWeights
) -> float:
    """Calculate weighted multi-objective score."""
    
    # Normalize metrics (assuming known ranges)
    normalized_completion = eval_result.completion_rate  # Already 0-1
    normalized_energy = 1.0 - min(eval_result.total_energy / 1000.0, 1.0)  # Invert
    normalized_risk = 1.0 - min(eval_result.total_exploss / 100.0, 1.0)    # Invert  
    normalized_time = 1.0 - min(eval_result.elapsed_time / 10.0, 1.0)      # Invert
    
    # Calculate weighted score
    score = (
        weights.completion_weight * normalized_completion +
        weights.energy_weight * normalized_energy +
        weights.risk_weight * normalized_risk +
        weights.time_weight * normalized_time
    )
    
    return score

def example_multi_objective():
    # Create scenario with varied UAV capabilities
    uavs = []
    for i in range(20):
        # Create UAVs with different energy profiles
        if i < 7:  # Energy-efficient UAVs
            uav = UAV(
                id=i, position=[np.random.uniform(0, 50), np.random.uniform(0, 50), 0],
                resources=[np.random.uniform(8, 12) for _ in range(4)],
                value=100, max_speed=20,
                fly_energy_per_time=1.0, hover_energy_per_time=0.5
            )
        elif i < 14:  # Fast UAVs
            uav = UAV(
                id=i, position=[np.random.uniform(0, 50), np.random.uniform(0, 50), 0],
                resources=[np.random.uniform(6, 10) for _ in range(4)],
                value=90, max_speed=35,
                fly_energy_per_time=3.0, hover_energy_per_time=2.0
            )
        else:  # High-capacity UAVs
            uav = UAV(
                id=i, position=[np.random.uniform(0, 50), np.random.uniform(0, 50), 0],
                resources=[np.random.uniform(15, 20) for _ in range(4)],
                value=120, max_speed=25,
                fly_energy_per_time=2.0, hover_energy_per_time=1.5
            )
        uavs.append(uav)
    
    # Create tasks with different risk levels
    tasks = []
    for i in range(8):
        threat_level = 0.1 if i < 3 else (0.5 if i < 6 else 0.8)  # Low, medium, high risk
        task = Task(
            id=i,
            position=[np.random.uniform(200, 800), np.random.uniform(200, 800), 0],
            required_resources=[np.random.uniform(15, 25) for _ in range(4)],
            time_window=[0, 50],
            threat=threat_level
        )
        tasks.append(task)
    
    # Test different objective weight configurations
    weight_configs = [
        ("Completion-focused", MultiObjectiveWeights(0.7, 0.1, 0.1, 0.1)),
        ("Energy-focused", MultiObjectiveWeights(0.3, 0.5, 0.1, 0.1)),
        ("Risk-averse", MultiObjectiveWeights(0.3, 0.2, 0.4, 0.1)),
        ("Time-critical", MultiObjectiveWeights(0.3, 0.2, 0.2, 0.3)),
        ("Balanced", MultiObjectiveWeights(0.25, 0.25, 0.25, 0.25)),
    ]
    
    results = {}
    
    print("=== Multi-Objective Optimization Analysis ===")
    
    for config_name, weights in weight_configs:
        print(f"\nTesting {config_name} configuration...")
        
        # Create fresh managers
        uav_manager = UAVManager(uavs.copy())
        task_manager = TaskManager(tasks.copy())
        coalition_manager = CoalitionManager(
            [uav.id for uav in uavs], [task.id for task in tasks]
        )
        
        # Configure hyperparameters based on objectives
        hyper_params = HyperParams(max_iter=50)
        
        # Adjust algorithm weights based on objectives
        if hasattr(hyper_params, 'alpha'):  # Resource weight
            hyper_params.alpha = 0.6 if weights.completion_weight > 0.5 else 0.4
        if hasattr(hyper_params, 'beta'):   # Distance weight (affects energy/time)
            hyper_params.beta = 0.4 if weights.energy_weight > 0.3 else 0.2
        if hasattr(hyper_params, 'gamma'):  # Threat weight
            hyper_params.gamma = 0.3 if weights.risk_weight > 0.3 else 0.1
        
        # Run solver
        solver = ChinaScience2024_CoalitionFormationGame(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
        
        start_time = time.time()
        result_coalition_manager = solver.run_allocate()
        elapsed_time = time.time() - start_time
        
        # Calculate detailed metrics
        completion_rate = result_coalition_manager.get_completion_rate()
        
        # Estimate energy consumption (simplified)
        total_energy = 0
        total_distance = 0
        total_risk = 0
        
        for task_id in task_manager.get_ids():
            coalition = result_coalition_manager.get_task_coalition(task_id)
            if coalition:
                task = task_manager.get(task_id)
                for uav_id in coalition:
                    uav = uav_manager.get(uav_id)
                    distance = uav.position.distance_to(task.position)
                    energy = (distance / uav.max_speed) * uav.fly_energy_per_time
                    total_energy += energy
                    total_distance += distance
                    total_risk += task.threat * uav.value
        
        # Create evaluation result
        eval_result = EvalResult(
            completion_rate=completion_rate,
            total_energy=total_energy,
            total_distance=total_distance,
            total_exploss=total_risk,
            elapsed_time=elapsed_time
        )
        
        # Calculate multi-objective score
        mo_score = calculate_multi_objective_score(eval_result, weights)
        
        results[config_name] = {
            "eval_result": eval_result,
            "mo_score": mo_score,
            "weights": weights
        }
        
        print(f"  Completion rate: {completion_rate:.2%}")
        print(f"  Total energy: {total_energy:.1f}")
        print(f"  Total risk: {total_risk:.1f}")
        print(f"  Execution time: {elapsed_time:.3f}s")
        print(f"  Multi-objective score: {mo_score:.3f}")
    
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]["mo_score"])
    
    print(f"\n=== Best Configuration ===")
    print(f"Configuration: {best_config[0]}")
    print(f"Multi-objective score: {best_config[1]['mo_score']:.3f}")
    best_config[1]["eval_result"].format_print()
    
    return results

if __name__ == "__main__":
    example_multi_objective()
```

---

## Research and Benchmarking

### Example 6: Comprehensive Benchmark Suite

Create a standardized benchmark for comparing algorithms.

```python
#!/usr/bin/env python3
"""
Example 6: Comprehensive Benchmark Suite
Standardized benchmark for algorithm comparison and research.
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Type
import matplotlib.pyplot as plt
import pandas as pd

from framework import UAVManager, TaskManager, CoalitionManager, HyperParams
from framework.utils import EvalResult
from solvers import *

@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result."""
    solver_name: str
    problem_size: str
    completion_rate: float
    resource_utilization: float
    total_distance: float
    total_energy: float
    execution_time: float
    convergence_iterations: int
    memory_usage: float  # MB
    success: bool
    error_message: str = ""

class BenchmarkSuite:
    """Comprehensive benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define test scenarios
        self.test_scenarios = [
            {"name": "small", "uav_nums": [5, 8, 10], "task_nums": [3, 5, 8]},
            {"name": "medium", "uav_nums": [15, 20, 25], "task_nums": [8, 12, 15]},
            {"name": "large", "uav_nums": [30, 40, 50], "task_nums": [15, 20, 25]},
        ]
        
        # Define solvers to test
        self.solvers = [
            ("CSCI2024", ChinaScience2024_CoalitionFormationGame),
            ("IROS2024", IROS2024_CoalitionFormationGame),
            ("ICRA2024", ICRA2024_CoalitionFormationGame),
            ("Auction", AuctionBiddingSolver),
            ("Distributed", DistributedSolver),
            ("MILP", MILPSolver),
            ("NLP", NLPSolverScipy),
        ]
    
    def generate_test_case(self, uav_num: int, task_num: int, seed: int = 42) -> tuple:
        """Generate standardized test case."""
        np.random.seed(seed)
        
        # Generate UAVs
        uavs = []
        for i in range(uav_num):
            uav = UAV(
                id=i,
                position=[np.random.uniform(0, 100), np.random.uniform(0, 100), 0],
                resources=[np.random.uniform(8, 15) for _ in range(5)],
                value=np.random.uniform(80, 120),
                max_speed=np.random.uniform(20, 35)
            )
            uavs.append(uav)
        
        # Generate tasks
        tasks = []
        for i in range(task_num):
            task = Task(
                id=i,
                position=[np.random.uniform(200, 800), np.random.uniform(200, 800), 0],
                required_resources=[np.random.uniform(20, 35) for _ in range(5)],
                time_window=[0, 60],
                threat=np.random.uniform(0.1, 0.6)
            )
            tasks.append(task)
        
        return uavs, tasks
    
    def run_single_benchmark(
        self, 
        solver_name: str, 
        solver_class: Type,
        uav_num: int, 
        task_num: int,
        repetitions: int = 3
    ) -> List[BenchmarkResult]:
        """Run benchmark for single solver configuration."""
        
        results = []
        
        for rep in range(repetitions):
            # Generate test case
            uavs, tasks = self.generate_test_case(uav_num, task_num, seed=rep)
            
            # Create managers
            uav_manager = UAVManager(uavs)
            task_manager = TaskManager(tasks)
            coalition_manager = CoalitionManager(
                [uav.id for uav in uavs], [task.id for task in tasks]
            )
            
            # Configure hyperparameters
            hyper_params = HyperParams(max_iter=100, tolerance=1e-4)
            
            # Skip expensive solvers for large problems
            if solver_name in ["MILP", "NLP"] and (uav_num > 25 or task_num > 15):
                result = BenchmarkResult(
                    solver_name=solver_name,
                    problem_size=f"{uav_num}x{task_num}",
                    completion_rate=0,
                    resource_utilization=0,
                    total_distance=0,
                    total_energy=0,
                    execution_time=0,
                    convergence_iterations=0,
                    memory_usage=0,
                    success=False,
                    error_message="Skipped - problem too large"
                )
                results.append(result)
                continue
            
            try:
                # Create solver
                solver = solver_class(uav_manager, task_manager, coalition_manager, hyper_params)
                
                # Run with timing and memory monitoring
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                start_time = time.time()
                result_coalition_manager = solver.run_allocate()
                execution_time = time.time() - start_time
                
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = memory_after - memory_before
                
                # Calculate metrics
                completion_rate = result_coalition_manager.get_completion_rate()
                resource_utilization = self._calculate_resource_utilization(
                    result_coalition_manager, uav_manager, task_manager
                )
                
                total_distance, total_energy = self._calculate_distance_energy(
                    result_coalition_manager, uav_manager, task_manager
                )
                
                result = BenchmarkResult(
                    solver_name=solver_name,
                    problem_size=f"{uav_num}x{task_num}",
                    completion_rate=completion_rate,
                    resource_utilization=resource_utilization,
                    total_distance=total_distance,
                    total_energy=total_energy,
                    execution_time=execution_time,
                    convergence_iterations=getattr(solver, 'iterations', 0),
                    memory_usage=memory_usage,
                    success=True
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    solver_name=solver_name,
                    problem_size=f"{uav_num}x{task_num}",
                    completion_rate=0,
                    resource_utilization=0,
                    total_distance=0,
                    total_energy=0,
                    execution_time=0,
                    convergence_iterations=0,
                    memory_usage=0,
                    success=False,
                    error_message=str(e)
                )
            
            results.append(result)
            print(f"  Rep {rep+1}/{repetitions}: {result.success}")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        
        all_results = []
        
        print("=== Running Comprehensive Benchmark Suite ===")
        
        for scenario in self.test_scenarios:
            print(f"\nTesting {scenario['name']} scenarios...")
            
            for uav_num in scenario['uav_nums']:
                for task_num in scenario['task_nums']:
                    problem_name = f"{scenario['name']}_{uav_num}x{task_num}"
                    print(f"\nProblem: {problem_name}")
                    
                    for solver_name, solver_class in self.solvers:
                        print(f"  Testing {solver_name}...")
                        
                        solver_results = self.run_single_benchmark(
                            solver_name, solver_class, uav_num, task_num
                        )
                        all_results.extend(solver_results)
        
        # Save raw results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        
        # Generate analysis
        self._generate_analysis(all_results)
        
        return {"results": all_results, "analysis_dir": self.output_dir}
    
    def _calculate_resource_utilization(self, coalition_manager, uav_manager, task_manager):
        """Calculate resource utilization rate."""
        total_available = 0
        total_used = 0
        
        for uav_id in uav_manager.get_ids():
            uav = uav_manager.get(uav_id)
            total_available += np.sum(uav.resources)
            
            # Check if UAV is assigned
            assigned_task = coalition_manager.get_uav_assignment(uav_id)
            if assigned_task is not None:
                total_used += np.sum(uav.resources)
        
        return total_used / total_available if total_available > 0 else 0
    
    def _calculate_distance_energy(self, coalition_manager, uav_manager, task_manager):
        """Calculate total distance and energy consumption."""
        total_distance = 0
        total_energy = 0
        
        for task_id in task_manager.get_ids():
            coalition = coalition_manager.get_task_coalition(task_id)
            if coalition:
                task = task_manager.get(task_id)
                for uav_id in coalition:
                    uav = uav_manager.get(uav_id)
                    distance = uav.position.distance_to(task.position)
                    energy = (distance / uav.max_speed) * uav.fly_energy_per_time
                    total_distance += distance
                    total_energy += energy
        
        return total_distance, total_energy
    
    def _generate_analysis(self, results: List[BenchmarkResult]):
        """Generate comprehensive analysis and plots."""
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(r) for r in results])
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            print("No successful results to analyze")
            return
        
        # Generate performance comparison plots
        self._plot_performance_comparison(successful_df)
        self._plot_scalability_analysis(successful_df)
        self._plot_efficiency_analysis(successful_df)
        
        # Generate summary statistics
        self._generate_summary_statistics(successful_df)
    
    def _plot_performance_comparison(self, df):
        """Generate performance comparison plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Completion rate by solver
        completion_by_solver = df.groupby('solver_name')['completion_rate'].mean()
        axes[0,0].bar(completion_by_solver.index, completion_by_solver.values)
        axes[0,0].set_title('Average Completion Rate by Solver')
        axes[0,0].set_ylabel('Completion Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Execution time by solver
        time_by_solver = df.groupby('solver_name')['execution_time'].mean()
        axes[0,1].bar(time_by_solver.index, time_by_solver.values)
        axes[0,1].set_title('Average Execution Time by Solver')
        axes[0,1].set_ylabel('Time (s)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Resource utilization by solver
        util_by_solver = df.groupby('solver_name')['resource_utilization'].mean()
        axes[1,0].bar(util_by_solver.index, util_by_solver.values)
        axes[1,0].set_title('Average Resource Utilization by Solver')
        axes[1,0].set_ylabel('Utilization Rate')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Energy efficiency by solver
        energy_by_solver = df.groupby('solver_name')['total_energy'].mean()
        axes[1,1].bar(energy_by_solver.index, energy_by_solver.values)
        axes[1,1].set_title('Average Energy Consumption by Solver')
        axes[1,1].set_ylabel('Energy')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300)
        plt.close()
    
    def _plot_scalability_analysis(self, df):
        """Generate scalability analysis plots."""
        
        # Extract problem sizes
        df['uav_num'] = df['problem_size'].str.split('x').str[0].astype(int)
        df['task_num'] = df['problem_size'].str.split('x').str[1].astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Completion rate vs UAV number
        for solver in df['solver_name'].unique():
            solver_df = df[df['solver_name'] == solver]
            uav_completion = solver_df.groupby('uav_num')['completion_rate'].mean()
            axes[0,0].plot(uav_completion.index, uav_completion.values, 
                          marker='o', label=solver)
        
        axes[0,0].set_xlabel('Number of UAVs')
        axes[0,0].set_ylabel('Completion Rate')
        axes[0,0].set_title('Scalability: Completion Rate vs UAV Number')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Execution time vs problem size
        df['total_entities'] = df['uav_num'] + df['task_num']
        for solver in df['solver_name'].unique():
            solver_df = df[df['solver_name'] == solver]
            size_time = solver_df.groupby('total_entities')['execution_time'].mean()
            axes[0,1].plot(size_time.index, size_time.values, 
                          marker='s', label=solver)
        
        axes[0,1].set_xlabel('Total Entities (UAVs + Tasks)')
        axes[0,1].set_ylabel('Execution Time (s)')
        axes[0,1].set_title('Scalability: Execution Time vs Problem Size')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scalability_analysis.png', dpi=300)
        plt.close()
    
    def _generate_summary_statistics(self, df):
        """Generate summary statistics."""
        
        summary_stats = df.groupby('solver_name').agg({
            'completion_rate': ['mean', 'std', 'min', 'max'],
            'execution_time': ['mean', 'std', 'min', 'max'],
            'resource_utilization': ['mean', 'std'],
            'total_energy': ['mean', 'std'],
            'memory_usage': ['mean', 'std']
        }).round(4)
        
        # Save to CSV
        summary_stats.to_csv(self.output_dir / 'summary_statistics.csv')
        
        # Generate text report
        with open(self.output_dir / 'benchmark_report.txt', 'w') as f:
            f.write("=== Comprehensive Benchmark Report ===\n\n")
            f.write(f"Total test cases: {len(df)}\n")
            f.write(f"Successful runs: {len(df[df['success']])}\n")
            f.write(f"Failed runs: {len(df[~df['success']])}\n\n")
            
            f.write("=== Performance Rankings ===\n")
            
            # Rank by completion rate
            completion_ranking = df.groupby('solver_name')['completion_rate'].mean().sort_values(ascending=False)
            f.write("\nCompletion Rate Ranking:\n")
            for i, (solver, rate) in enumerate(completion_ranking.items(), 1):
                f.write(f"{i}. {solver}: {rate:.3f}\n")
            
            # Rank by speed
            speed_ranking = df.groupby('solver_name')['execution_time'].mean().sort_values()
            f.write("\nExecution Speed Ranking:\n")
            for i, (solver, time) in enumerate(speed_ranking.items(), 1):
                f.write(f"{i}. {solver}: {time:.3f}s\n")
            
            f.write(f"\n=== Detailed Statistics ===\n")
            f.write(summary_stats.to_string())

def example_comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    
    # Create and run benchmark suite
    benchmark = BenchmarkSuite(output_dir="benchmark_results_comprehensive")
    results = benchmark.run_full_benchmark()
    
    print(f"\n=== Benchmark Complete ===")
    print(f"Results saved to: {results['analysis_dir']}")
    print(f"Total test cases: {len(results['results'])}")
    
    # Print top performers
    successful_results = [r for r in results['results'] if r.success]
    if successful_results:
        # Best completion rate
        best_completion = max(successful_results, key=lambda x: x.completion_rate)
        print(f"\nBest completion rate: {best_completion.solver_name} "
              f"({best_completion.completion_rate:.2%})")
        
        # Fastest solver
        fastest = min(successful_results, key=lambda x: x.execution_time)
        print(f"Fastest solver: {fastest.solver_name} "
              f"({fastest.execution_time:.3f}s)")
    
    return results

if __name__ == "__main__":
    example_comprehensive_benchmark()
```

---

## Custom Development

### Example 7: Custom Solver Development

Create your own solver algorithm.

```python
#!/usr/bin/env python3
"""
Example 7: Custom Solver Development
Demonstrates how to implement a custom MRTA solver.
"""

from framework.mrta_solver import MRTASolver
from framework import UAVManager, TaskManager, CoalitionManager, HyperParams
import numpy as np
import random
from typing import List, Dict, Tuple

class GreedyResourceSolver(MRTASolver):
    """
    Custom greedy solver that prioritizes resource matching.
    
    Algorithm:
    1. Sort tasks by resource requirements (descending)
    2. For each task, find the best UAV coalition using greedy selection
    3. UAVs are selected based on resource contribution and distance
    """
    
    def __init__(self, uav_manager, task_manager, coalition_manager, hyper_params):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        
        # Custom parameters
        self.resource_weight = 0.7
        self.distance_weight = 0.2
        self.diversity_weight = 0.1
        self.max_coalition_size = 5
    
    @classmethod
    def type_name(cls):
        return "GreedyResource"
    
    def run_allocate(self, init_method: str = "random"):
        """Run the greedy resource allocation algorithm."""
        
        print(f"Running {self.type_name()} allocation...")
        
        # Clear existing assignments
        self.coalition_manager.clear_all_assignments()
        
        # Get all tasks sorted by total resource requirements
        tasks = self.task_manager.get_all()
        tasks.sort(key=lambda t: np.sum(t.required_resources), reverse=True)
        
        # Track available UAVs
        available_uavs = set(self.uav_manager.get_ids())
        
        for task in tasks:
            print(f"  Assigning task {task.id} (req: {task.required_resources})")
            
            # Find best coalition for this task
            best_coalition = self._find_best_coalition(task, available_uavs)
            
            if best_coalition:
                # Assign coalition to task
                self.coalition_manager.assign_coalition_to_task(task.id, best_coalition)
                
                # Remove assigned UAVs from available set
                available_uavs -= set(best_coalition)
                
                print(f"    Assigned UAVs: {best_coalition}")
                
                # Calculate resource satisfaction
                total_resources = self.uav_manager.get_available_resources(best_coalition)
                satisfaction = np.minimum(total_resources / task.required_resources, 1.0)
                print(f"    Resource satisfaction: {satisfaction.mean():.2f}")
            else:
                print(f"    No suitable coalition found")
        
        return self.coalition_manager
    
    def _find_best_coalition(self, task, available_uavs: set) -> List[int]:
        """Find the best UAV coalition for a task using greedy selection."""
        
        if not available_uavs:
            return []
        
        coalition = []
        remaining_resources = task.required_resources.copy()
        candidate_uavs = list(available_uavs)
        
        # Greedy selection process
        for _ in range(min(self.max_coalition_size, len(candidate_uavs))):
            if np.all(remaining_resources <= 0):
                break  # Task requirements satisfied
            
            best_uav = None
            best_score = -float('inf')
            
            for uav_id in candidate_uavs:
                if uav_id in coalition:
                    continue
                    
                uav = self.uav_manager.get(uav_id)
                score = self._calculate_uav_score(uav, task, remaining_resources, coalition)
                
                if score > best_score:
                    best_score = score
                    best_uav = uav_id
            
            if best_uav is not None:
                coalition.append(best_uav)
                uav = self.uav_manager.get(best_uav)
                remaining_resources -= uav.resources
                remaining_resources = np.maximum(remaining_resources, 0)
            else:
                break
        
        # Only return coalition if it can meaningfully contribute
        if coalition:
            total_resources = self.uav_manager.get_available_resources(coalition)
            resource_match = np.sum(np.minimum(total_resources, task.required_resources))
            required_total = np.sum(task.required_resources)
            
            if resource_match / required_total >= 0.5:  # At least 50% match
                return coalition
        
        return []
    
    def _calculate_uav_score(self, uav, task, remaining_resources, current_coalition):
        """Calculate score for adding UAV to coalition."""
        
        # Resource contribution score
        resource_contribution = np.sum(np.minimum(uav.resources, remaining_resources))
        resource_score = resource_contribution / np.sum(remaining_resources) if np.sum(remaining_resources) > 0 else 0
        
        # Distance score (closer is better)
        distance = uav.position.distance_to(task.position)
        max_distance = 1000  # Normalize by expected max distance
        distance_score = max(0, 1 - distance / max_distance)
        
        # Diversity score (prefer different resource profiles)
        diversity_score = 0
        if current_coalition:
            coalition_resources = self.uav_manager.get_available_resources(current_coalition)
            # Prefer UAVs with different resource strengths
            resource_similarity = np.corrcoef(uav.resources, coalition_resources)[0, 1]
            diversity_score = max(0, 1 - abs(resource_similarity))
        else:
            diversity_score = 1.0  # First UAV gets full diversity score
        
        # Combined score
        total_score = (
            self.resource_weight * resource_score +
            self.distance_weight * distance_score +
            self.diversity_weight * diversity_score
        )
        
        return total_score

class AdaptiveGeneticSolver(MRTASolver):
    """
    Custom genetic algorithm solver for MRTA.
    
    Uses evolutionary approach to optimize task assignments.
    """
    
    def __init__(self, uav_manager, task_manager, coalition_manager, hyper_params):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        
        # Genetic algorithm parameters
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        self.max_generations = 100
    
    @classmethod
    def type_name(cls):
        return "AdaptiveGenetic"
    
    def run_allocate(self, init_method: str = "random"):
        """Run genetic algorithm for task allocation."""
        
        print(f"Running {self.type_name()} allocation...")
        
        # Initialize population
        population = self._initialize_population()
        
        best_fitness = -float('inf')
        generations_without_improvement = 0
        
        for generation in range(self.max_generations):
            # Evaluate fitness for all individuals
            fitness_scores = [self._evaluate_fitness(individual) for individual in population]
            
            # Track best solution
            current_best = max(fitness_scores)
            if current_best > best_fitness:
                best_fitness = current_best
                generations_without_improvement = 0
                best_individual = population[fitness_scores.index(current_best)]
            else:
                generations_without_improvement += 1
            
            print(f"  Generation {generation}: Best fitness = {current_best:.3f}")
            
            # Early stopping
            if generations_without_improvement >= 20:
                print(f"  Converged after {generation} generations")
                break
            
            # Create new population
            population = self._evolve_population(population, fitness_scores)
        
        # Apply best solution
        self._apply_solution(best_individual)
        
        return self.coalition_manager
    
    def _initialize_population(self) -> List[Dict]:
        """Initialize random population of solutions."""
        
        population = []
        uav_ids = self.uav_manager.get_ids()
        task_ids = self.task_manager.get_ids()
        
        for _ in range(self.population_size):
            individual = {}
            available_uavs = set(uav_ids)
            
            for task_id in task_ids:
                # Randomly assign 1-4 UAVs to task (or none)
                if available_uavs and random.random() < 0.8:  # 80% chance to assign
                    coalition_size = random.randint(1, min(4, len(available_uavs)))
                    coalition = random.sample(list(available_uavs), coalition_size)
                    individual[task_id] = coalition
                    available_uavs -= set(coalition)
                else:
                    individual[task_id] = []
            
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual: Dict) -> float:
        """Evaluate fitness of an individual solution."""
        
        total_fitness = 0
        
        for task_id, coalition in individual.items():
            if not coalition:
                continue
                
            task = self.task_manager.get(task_id)
            
            # Resource satisfaction
            total_resources = self.uav_manager.get_available_resources(coalition)
            resource_satisfaction = np.minimum(total_resources / task.required_resources, 1.0)
            resource_score = np.mean(resource_satisfaction)
            
            # Distance penalty
            distances = [self.uav_manager.get(uav_id).position.distance_to(task.position) 
                        for uav_id in coalition]
            distance_penalty = np.mean(distances) / 1000.0  # Normalize
            
            # Coalition efficiency (prefer smaller coalitions if possible)
            efficiency_bonus = 1.0 / len(coalition) if len(coalition) > 0 else 0
            
            # Task fitness
            task_fitness = resource_score - 0.3 * distance_penalty + 0.1 * efficiency_bonus
            total_fitness += max(0, task_fitness)
        
        return total_fitness
    
    def _evolve_population(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Evolve population using genetic operators."""
        
        new_population = []
        
        # Elitism: keep best individuals
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], tournament_size: int = 3) -> Dict:
        """Select individual using tournament selection."""
        
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Perform crossover between two parents."""
        
        child1, child2 = {}, {}
        task_ids = list(parent1.keys())
        
        # Single-point crossover
        crossover_point = random.randint(0, len(task_ids))
        
        for i, task_id in enumerate(task_ids):
            if i < crossover_point:
                child1[task_id] = parent1[task_id].copy()
                child2[task_id] = parent2[task_id].copy()
            else:
                child1[task_id] = parent2[task_id].copy()
                child2[task_id] = parent1[task_id].copy()
        
        return child1, child2
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutate an individual."""
        
        mutated = individual.copy()
        task_ids = list(mutated.keys())
        uav_ids = self.uav_manager.get_ids()
        
        # Random mutation: modify one task's assignment
        task_id = random.choice(task_ids)
        
        if random.random() < 0.5:
            # Add/remove UAV from coalition
            current_coalition = set(mutated[task_id])
            available_uavs = set(uav_ids) - set().union(*[set(coal) for coal in mutated.values()])
            available_uavs |= current_coalition  # Current coalition UAVs are available
            
            if current_coalition and random.random() < 0.5:
                # Remove UAV
                uav_to_remove = random.choice(list(current_coalition))
                current_coalition.remove(uav_to_remove)
            elif available_uavs:
                # Add UAV
                uav_to_add = random.choice(list(available_uavs - current_coalition))
                current_coalition.add(uav_to_add)
            
            mutated[task_id] = list(current_coalition)
        else:
            # Completely reassign task
            available_uavs = set(uav_ids) - set().union(*[set(coal) for t, coal in mutated.items() if t != task_id])
            if available_uavs:
                coalition_size = random.randint(0, min(3, len(available_uavs)))
                if coalition_size > 0:
                    mutated[task_id] = random.sample(list(available_uavs), coalition_size)
                else:
                    mutated[task_id] = []
        
        return mutated
    
    def _apply_solution(self, solution: Dict):
        """Apply the best solution to coalition manager."""
        
        self.coalition_manager.clear_all_assignments()
        
        for task_id, coalition in solution.items():
            if coalition:
                self.coalition_manager.assign_coalition_to_task(task_id, coalition)

def example_custom_solver():
    """Test custom solvers."""
    
    # Generate test scenario
    from gen import gen_iros2024_case
    import json
    
    # Generate test case
    gen_iros2024_case("test_cases/custom_solver_test.json")
    
    # Load test case
    with open("test_cases/custom_solver_test.json", "r") as f:
        data = json.load(f)
    
    uavs = [UAV.from_dict(uav_data) for uav_data in data["uavs"]]
    tasks = [Task.from_dict(task_data) for task_data in data["tasks"]]
    
    # Test custom solvers
    custom_solvers = [
        ("Greedy Resource", GreedyResourceSolver),
        ("Adaptive Genetic", AdaptiveGeneticSolver),
    ]
    
    print("=== Testing Custom Solvers ===")
    
    results = {}
    
    for solver_name, solver_class in custom_solvers:
        print(f"\nTesting {solver_name}...")
        
        # Create fresh managers
        uav_manager = UAVManager(uavs.copy())
        task_manager = TaskManager(tasks.copy())
        coalition_manager = CoalitionManager(
            [uav.id for uav in uavs], [task.id for task in tasks]
        )
        
        hyper_params = HyperParams(max_iter=50)
        
        # Create and run solver
        solver = solver_class(uav_manager, task_manager, coalition_manager, hyper_params)
        
        start_time = time.time()
        result_coalition_manager = solver.run_allocate()
        elapsed_time = time.time() - start_time
        
        # Evaluate results
        completion_rate = result_coalition_manager.get_completion_rate()
        
        results[solver_name] = {
            "completion_rate": completion_rate,
            "elapsed_time": elapsed_time,
            "assignments": result_coalition_manager.get_assignment_summary()
        }
        
        print(f"✓ Completion rate: {completion_rate:.2%}")
        print(f"✓ Execution time: {elapsed_time:.3f}s")
    
    return results

if __name__ == "__main__":
    example_custom_solver()
```

---

## Performance Optimization

### Example 8: Performance Optimization Techniques

Optimize solver performance for large-scale problems.

```python
#!/usr/bin/env python3
"""
Example 8: Performance Optimization Techniques
Demonstrates various techniques to optimize solver performance.
"""

import time
import cProfile
import pstats
from memory_profiler import profile
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

from framework import UAV, Task, UAVManager, TaskManager, CoalitionManager, HyperParams
from solvers import IROS2024_CoalitionFormationGame
import json

class OptimizedSolver(IROS2024_CoalitionFormationGame):
    """Optimized version of IROS2024 solver with performance improvements."""
    
    def __init__(self, uav_manager, task_manager, coalition_manager, hyper_params):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        
        # Pre-compute expensive operations
        self._precompute_distances()
        self._precompute_resource_matrices()
        
        # Enable caching
        self._cache = {}
        self._enable_caching = True
    
    def _precompute_distances(self):
        """Pre-compute all UAV-task distances."""
        
        self.distance_matrix = {}
        
        for uav in self.uav_manager.get_all():
            self.distance_matrix[uav.id] = {}
            for task in self.task_manager.get_all():
                distance = uav.position.distance_to(task.position)
                self.distance_matrix[uav.id][task.id] = distance
    
    def _precompute_resource_matrices(self):
        """Pre-compute resource compatibility matrices."""
        
        self.resource_compatibility = {}
        
        for task in self.task_manager.get_all():
            self.resource_compatibility[task.id] = {}
            
            for uav in self.uav_manager.get_all():
                # Calculate resource contribution score
                contribution = np.minimum(uav.resources, task.required_resources)
                score = np.sum(contribution) / np.sum(task.required_resources)
                self.resource_compatibility[task.id][uav.id] = score
    
    def get_distance(self, uav_id: int, task_id: int) -> float:
        """Get pre-computed distance."""
        return self.distance_matrix[uav_id][task_id]
    
    def get_resource_compatibility(self, task_id: int, uav_id: int) -> float:
        """Get pre-computed resource compatibility score."""
        return self.resource_compatibility[task_id][uav_id]

def performance_profiling_example():
    """Profile solver performance to identify bottlenecks."""
    
    print("=== Performance Profiling Example ===")
    
    # Generate large test case
    uavs = []
    for i in range(100):
        uav = UAV(
            id=i,
            position=[np.random.uniform(0, 1000), np.random.uniform(0, 1000), 0],
            resources=[np.random.uniform(5, 15) for _ in range(5)],
            value=np.random.uniform(80, 120),
            max_speed=np.random.uniform(20, 35)
        )
        uavs.append(uav)
    
    tasks = []
    for i in range(30):
        task = Task(
            id=i,
            position=[np.random.uniform(0, 1000), np.random.uniform(0, 1000), 0],
            required_resources=[np.random.uniform(20, 40) for _ in range(5)],
            time_window=[0, 100],
            threat=np.random.uniform(0.1, 0.6)
        )
        tasks.append(task)
    
    # Create managers
    uav_manager = UAVManager(uavs)
    task_manager = TaskManager(tasks)
    coalition_manager = CoalitionManager(
        [uav.id for uav in uavs], [task.id for task in tasks]
    )
    hyper_params = HyperParams(max_iter=20)
    
    # Profile standard solver
    print("Profiling standard solver...")
    
    def run_standard_solver():
        solver = IROS2024_CoalitionFormationGame(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
        return solver.run_allocate()
    
    # Run with profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    result1 = run_standard_solver()
    standard_time = time.time() - start_time
    
    profiler.disable()
    
    # Save profiling results
    profiler.dump_stats('standard_solver_profile.prof')
    
    # Print top time-consuming functions
    stats = pstats.Stats('standard_solver_profile.prof')
    print("\nTop 10 time-consuming functions (standard solver):")
    stats.sort_stats('cumulative').print_stats(10)
    
    # Profile optimized solver
    print("\nProfiling optimized solver...")
    
    def run_optimized_solver():
        solver = OptimizedSolver(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
        return solver.run_allocate()
    
    start_time = time.time()
    result2 = run_optimized_solver()
    optimized_time = time.time() - start_time
    
    # Compare results
    print(f"\n=== Performance Comparison ===")
    print(f"Standard solver time: {standard_time:.3f}s")
    print(f"Optimized solver time: {optimized_time:.3f}s")
    print(f"Speedup: {standard_time/optimized_time:.2f}x")
    
    completion1 = result1.get_completion_rate()
    completion2 = result2.get_completion_rate()
    print(f"Standard completion rate: {completion1:.2%}")
    print(f"Optimized completion rate: {completion2:.2%}")
    
    return {
        "standard_time": standard_time,
        "optimized_time": optimized_time,
        "speedup": standard_time/optimized_time
    }

@profile
def memory_profiling_example():
    """Profile memory usage of solver."""
    
    print("=== Memory Profiling Example ===")
    
    # Generate test case
    uavs = []
    for i in range(50):
        uav = UAV(
            id=i,
            position=[np.random.uniform(0, 500), np.random.uniform(0, 500), 0],
            resources=[np.random.uniform(8, 12) for _ in range(5)],
            value=100,
            max_speed=25
        )
        uavs.append(uav)
    
    tasks = []
    for i in range(20):
        task = Task(
            id=i,
            position=[np.random.uniform(0, 500), np.random.uniform(0, 500), 0],
            required_resources=[np.random.uniform(15, 25) for _ in range(5)],
            time_window=[0, 50],
            threat=0.3
        )
        tasks.append(task)
    
    # Create managers
    uav_manager = UAVManager(uavs)
    task_manager = TaskManager(tasks)
    coalition_manager = CoalitionManager(
        [uav.id for uav in uavs], [task.id for task in tasks]
    )
    hyper_params = HyperParams(max_iter=30)
    
    # Run solver with memory profiling
    solver = IROS2024_CoalitionFormationGame(
        uav_manager, task_manager, coalition_manager, hyper_params
    )
    
    result = solver.run_allocate()
    
    return result

def parallel_processing_example():
    """Demonstrate parallel processing for multiple test cases."""
    
    print("=== Parallel Processing Example ===")
    
    def run_single_test(test_params):
        """Run a single test case."""
        uav_num, task_num, seed = test_params
        
        np.random.seed(seed)
        
        # Generate test case
        uavs = []
        for i in range(uav_num):
            uav = UAV(
                id=i,
                position=[np.random.uniform(0, 100), np.random.uniform(0, 100), 0],
                resources=[np.random.uniform(8, 12) for _ in range(5)],
                value=100,
                max_speed=25
            )
            uavs.append(uav)
        
        tasks = []
        for i in range(task_num):
            task = Task(
                id=i,
                position=[np.random.uniform(200, 800), np.random.uniform(200, 800), 0],
                required_resources=[np.random.uniform(15, 25) for _ in range(5)],
                time_window=[0, 50],
                threat=0.3
            )
            tasks.append(task)
        
        # Create managers and run solver
        uav_manager = UAVManager(uavs)
        task_manager = TaskManager(tasks)
        coalition_manager = CoalitionManager(
            [uav.id for uav in uavs], [task.id for task in tasks]
        )
        hyper_params = HyperParams(max_iter=20)
        
        solver = IROS2024_CoalitionFormationGame(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
        
        start_time = time.time()
        result = solver.run_allocate()
        elapsed_time = time.time() - start_time
        
        completion_rate = result.get_completion_rate()
        
        return {
            "uav_num": uav_num,
            "task_num": task_num,
            "seed": seed,
            "completion_rate": completion_rate,
            "elapsed_time": elapsed_time
        }
    
    # Define test cases
    test_cases = [
        (20, 10, i) for i in range(5)  # 5 repetitions with 20 UAVs, 10 tasks
    ] + [
        (30, 15, i) for i in range(5)  # 5 repetitions with 30 UAVs, 15 tasks
    ]
    
    # Sequential execution
    print("Running sequential execution...")
    start_time = time.time()
    sequential_results = [run_single_test(params) for params in test_cases]
    sequential_time = time.time() - start_time
    
    # Parallel execution
    print("Running parallel execution...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        parallel_results = list(executor.map(run_single_test, test_cases))
    
    parallel_time = time.time() - start_time
    
    # Compare results
    print(f"\n=== Parallel Processing Results ===")
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Parallel time: {parallel_time:.3f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    print(f"Number of test cases: {len(test_cases)}")
    print(f"CPU cores used: {multiprocessing.cpu_count()}")
    
    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": sequential_time/parallel_time,
        "results": parallel_results
    }

def batch_optimization_example():
    """Demonstrate batch processing optimization techniques."""
    
    print("=== Batch Optimization Example ===")
    
    # Generate multiple test scenarios
    scenarios = []
    for scenario_id in range(10):
        uav_num = np.random.randint(20, 50)
        task_num = np.random.randint(8, 20)
        scenarios.append((scenario_id, uav_num, task_num))
    
    # Batch processing with shared resources
    print("Processing scenarios in batches...")
    
    batch_size = 3
    results = []
    
    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: scenarios {[s[0] for s in batch]}")
        
        batch_results = []
        
        for scenario_id, uav_num, task_num in batch:
            # Generate test case (reuse generation logic)
            uavs = []
            for j in range(uav_num):
                uav = UAV(
                    id=j,
                    position=[np.random.uniform(0, 100), np.random.uniform(0, 100), 0],
                    resources=[np.random.uniform(8, 12) for _ in range(5)],
                    value=100,
                    max_speed=25
                )
                uavs.append(uav)
            
            tasks = []
            for j in range(task_num):
                task = Task(
                    id=j,
                    position=[np.random.uniform(200, 800), np.random.uniform(200, 800), 0],
                    required_resources=[np.random.uniform(15, 25) for _ in range(5)],
                    time_window=[0, 50],
                    threat=0.3
                )
                tasks.append(task)
            
            # Run optimized solver
            uav_manager = UAVManager(uavs)
            task_manager = TaskManager(tasks)
            coalition_manager = CoalitionManager(
                [uav.id for uav in uavs], [task.id for task in tasks]
            )
            hyper_params = HyperParams(max_iter=15)  # Reduced iterations for speed
            
            solver = OptimizedSolver(
                uav_manager, task_manager, coalition_manager, hyper_params
            )
            
            start_time = time.time()
            result = solver.run_allocate()
            elapsed_time = time.time() - start_time
            
            batch_results.append({
                "scenario_id": scenario_id,
                "uav_num": uav_num,
                "task_num": task_num,
                "completion_rate": result.get_completion_rate(),
                "elapsed_time": elapsed_time
            })
        
        results.extend(batch_results)
        
        # Print batch summary
        avg_completion = np.mean([r["completion_rate"] for r in batch_results])
        avg_time = np.mean([r["elapsed_time"] for r in batch_results])
        print(f"  Batch average completion rate: {avg_completion:.2%}")
        print(f"  Batch average time: {avg_time:.3f}s")
    
    # Overall summary
    print(f"\n=== Batch Processing Summary ===")
    overall_avg_completion = np.mean([r["completion_rate"] for r in results])
    overall_avg_time = np.mean([r["elapsed_time"] for r in results])
    total_time = sum([r["elapsed_time"] for r in results])
    
    print(f"Processed {len(scenarios)} scenarios")
    print(f"Overall average completion rate: {overall_avg_completion:.2%}")
    print(f"Overall average time per scenario: {overall_avg_time:.3f}s")
    print(f"Total processing time: {total_time:.3f}s")
    
    return results

if __name__ == "__main__":
    # Run performance optimization examples
    print("Starting performance optimization examples...")
    
    # 1. Performance profiling
    profiling_results = performance_profiling_example()
    
    # 2. Memory profiling (requires memory_profiler package)
    try:
        memory_results = memory_profiling_example()
        print("Memory profiling completed successfully")
    except ImportError:
        print("Memory profiler not available. Install with: pip install memory_profiler")
    
    # 3. Parallel processing
    parallel_results = parallel_processing_example()
    
    # 4. Batch optimization
    batch_results = batch_optimization_example()
    
    print("\n=== All Performance Tests Completed ===")
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Solver Convergence Problems
```python
# Problem: Solver doesn't converge or takes too long

# Solution 1: Adjust hyperparameters
hyper_params = HyperParams(
    max_iter=200,      # Increase max iterations
    tolerance=1e-3,    # Relax convergence tolerance
    random_seed=42     # Use fixed seed for reproducibility
)

# Solution 2: Use different initialization
result = solver.run_allocate(init_method="greedy")  # Instead of "random"

# Solution 3: Check problem feasibility
def check_feasibility(uav_manager, task_manager):
    total_available = np.sum([np.sum(uav.resources) for uav in uav_manager.get_all()])
    total_required = np.sum([np.sum(task.required_resources) for task in task_manager.get_all()])
    
    if total_required > total_available:
        print(f"WARNING: Problem may be infeasible")
        print(f"Total required: {total_required}, Total available: {total_available}")
        return False
    return True
```

#### Issue 2: Memory Issues with Large Problems
```python
# Problem: Out of memory errors with large UAV swarms

# Solution 1: Use distributed solvers
from solvers import DistributedSolver
solver = DistributedSolver(uav_manager, task_manager, coalition_manager, hyper_params)

# Solution 2: Process in batches
def solve_in_batches(uav_manager, task_manager, batch_size=20):
    uav_batches = [uav_manager.get_all()[i:i+batch_size] 
                   for i in range(0, len(uav_manager.get_all()), batch_size)]
    
    results = []
    for batch in uav_batches:
        batch_manager = UAVManager(batch)
        # Process batch...
        results.append(batch_result)
    
    return combine_results(results)

# Solution 3: Use memory-efficient data structures
# Enable garbage collection
import gc
gc.collect()
```

#### Issue 3: Performance Issues
```python
# Problem: Solver runs too slowly

# Solution 1: Profile and optimize
import cProfile
profiler = cProfile.Profile()
profiler.enable()
result = solver.run_allocate()
profiler.disable()
profiler.dump_stats('profile.prof')

# Solution 2: Use faster solvers for large problems
if len(uav_manager.get_all()) > 50:
    solver = IROS2024_CoalitionFormationGame(...)  # Faster for large problems
else:
    solver = ChinaScience2024_CoalitionFormationGame(...)  # Better quality for small problems

# Solution 3: Parallel processing
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(solve_subproblem, subproblem) for subproblem in subproblems]
    results = [future.result() for future in futures]
```

This comprehensive usage guide provides practical examples for all major use cases of the Swarm UAV Task Assignment framework, from basic usage to advanced research scenarios and performance optimization techniques.