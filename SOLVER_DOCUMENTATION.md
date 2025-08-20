# MRTA Solver Algorithms - Detailed Documentation

## Overview

This document provides detailed documentation for all Multi-Robot Task Assignment (MRTA) solver algorithms implemented in the framework. Each solver implements different approaches to the coalition formation and task allocation problem.

## Table of Contents

- [Coalition Formation Game Solvers](#coalition-formation-game-solvers)
- [Optimization-Based Solvers](#optimization-based-solvers)
- [Auction-Based Solvers](#auction-based-solvers)
- [Distributed Solvers](#distributed-solvers)
- [Enumeration and Brute Force Solvers](#enumeration-and-brute-force-solvers)
- [Solver Comparison](#solver-comparison)
- [Performance Guidelines](#performance-guidelines)

---

## Coalition Formation Game Solvers

### ChinaScience2024_CoalitionFormationGame

**Paper Reference**: CSCI 2024 - "Coalition Formation Game for Multi-UAV Task Assignment"

**Algorithm Description**: 
Implements a coalition formation game where UAVs form coalitions to execute tasks. The game maximizes total system benefit through iterative coalition restructuring.

**Mathematical Model**:
- Game: G = (U, E, ε, R)
- U: Set of UAVs
- E: Task set (equivalent to T)
- ε: UAV utility function
- R: Coalition utility evaluation function

**Key Equations**:
```
R(ctj) = Σ(ui ∈ ctj) r(ui, tj)  # Coalition utility for task tj
SR = Σ(ctj ∈ CS) R(ctj)         # Total system revenue
r(ui, tj) = α × val(ui, tj) + β × dist(ui, tj) - γ × risk(ui, tj)  # UAV utility
ε(eu, E_{-ui}) = R(ctj) - R(ctj - {ui})  # UAV marginal contribution
```

**API Usage**:
```python
from solvers.csci2024 import ChinaScience2024_CoalitionFormationGame

# Initialize solver
solver = ChinaScience2024_CoalitionFormationGame(
    uav_manager=uav_manager,
    task_manager=task_manager,
    coalition_manager=coalition_manager,
    hyper_params=hyper_params
)

# Configure hyperparameters
hyper_params.max_iter = 50
hyper_params.tolerance = 1e-4
hyper_params.alpha = 0.6      # Resource weight
hyper_params.beta = 0.3       # Distance weight  
hyper_params.gamma = 0.1      # Threat weight

# Run allocation
result = solver.run_allocate(init_method="random")
```

**Constraints**:
1. Speed constraints
2. Task time window constraints
3. Single task per UAV constraint

**Best Use Cases**:
- Medium to large-scale problems (20-100 UAVs)
- Multi-resource tasks
- Time-sensitive operations
- When coalition formation is important

---

### IROS2024_CoalitionFormationGame

**Paper Reference**: IROS 2024 - "Distributed Coalition Formation for Multi-UAV Task Assignment"

**Algorithm Description**:
Implements a distributed coalition formation algorithm with leader-follower structure. UAVs discovering tasks become leaders, while others become followers that can switch coalitions.

**Algorithm Pseudocode**:
```cpp
Partition Alg1(R /* UAV set*/, T /* Tasks set */) {
    S = Partition(); // Initialize partition
    while (l < l_max) do {
        for (each ri in R, each tj in T) do {
            // Calculate utility values u1, u2, u3, u4
            if (u1 + u2 < u3 + u4) {
                // UAV ri switches coalition
            }
        }
        if (coalition structure S unchanged) {
            l = l + 1;
        } else {
            l = 0;
        }
    }
    return S;
}
```

**API Usage**:
```python
from solvers.iros2024 import IROS2024_CoalitionFormationGame

solver = IROS2024_CoalitionFormationGame(
    uav_manager, task_manager, coalition_manager, hyper_params
)

# Configure for distributed operation
hyper_params.max_iter = 30
hyper_params.convergence_threshold = 3  # l_max parameter
hyper_params.leader_selection = "discovery"  # or "random"

result = solver.run_allocate()
```

**Key Features**:
- Leader-follower structure
- Distributed decision making
- Coalition switching mechanism
- Convergence detection

**Best Use Cases**:
- Distributed UAV swarms
- Communication-constrained environments
- Real-time applications
- Scalable deployments

---

### ICRA2024_CoalitionFormationGame

**Paper Reference**: ICRA 2024 - "Multi-UAV Cooperative Task Assignment with Communication Constraints"

**Algorithm Description**:
Advanced coalition formation game with communication modeling and multi-process capability.

**API Usage**:
```python
from solvers.icra2024 import ICRA2024_CoalitionFormationGame

solver = ICRA2024_CoalitionFormationGame(
    uav_manager, task_manager, coalition_manager, hyper_params
)

# Communication-aware parameters
hyper_params.comm_range = 200.0  # Communication range in meters
hyper_params.bandwidth_limit = 10.0  # kHz
hyper_params.enable_multiprocess = True

result = solver.run_allocate()
```

**Special Features**:
- Communication range modeling
- Bandwidth constraints
- Multi-process simulation support
- Real-time execution capability

---

## Optimization-Based Solvers

### MILPSolver

**Description**: Mixed-Integer Linear Programming approach to MRTA problem.

**Mathematical Formulation**:
```
Minimize: Σ(i,j) cij × xij + Σ(i,j) dij × yij

Subject to:
- Σj xij ≤ 1, ∀i (UAV assignment constraint)
- Σi rij × xij ≥ Rj, ∀j (Resource requirement constraint)
- xij ∈ {0,1} (Binary assignment variables)
- yij ≥ 0 (Distance variables)
```

**API Usage**:
```python
from solvers.milp_solver import MILPSolver

solver = MILPSolver(uav_manager, task_manager, coalition_manager, hyper_params)

# MILP-specific parameters
hyper_params.solver_name = "CPLEX"  # or "GUROBI", "GLPK"
hyper_params.time_limit = 300  # seconds
hyper_params.mip_gap = 0.01  # 1% optimality gap

result = solver.run_allocate()
```

**Advantages**:
- Guaranteed optimal solution (within time limit)
- Handles complex constraints well
- Mature solver technology

**Limitations**:
- Exponential complexity
- Requires commercial solvers for large problems
- Less suitable for real-time applications

---

### NLPSolverScipy

**Description**: Non-Linear Programming solver using SciPy optimization.

**API Usage**:
```python
from solvers.nlp_solver import NLPSolverScipy

solver = NLPSolverScipy(uav_manager, task_manager, coalition_manager, hyper_params)

# NLP parameters
hyper_params.method = "SLSQP"  # Sequential Least Squares Programming
hyper_params.tolerance = 1e-6
hyper_params.max_iterations = 1000

result = solver.run_allocate()
```

**Supported Methods**:
- SLSQP: Sequential Least Squares Programming
- L-BFGS-B: Limited-memory BFGS with bounds
- TNC: Truncated Newton Algorithm

---

## Auction-Based Solvers

### AuctionBiddingSolver

**Description**: Enhanced auction mechanism supporting coalition formation through combinatorial bidding.

**Algorithm Features**:
- Combinatorial auctions for coalition formation
- Marginal utility bidding
- Bid decay mechanism
- Dynamic coalition sizing

**API Usage**:
```python
from solvers.auction_solver import AuctionBiddingSolver

solver = AuctionBiddingSolver(uav_manager, task_manager, coalition_manager, hyper_params)

# Auction parameters
solver.max_coalition_size = 5
solver.bid_decay_factor = 0.95
solver.min_bid_improvement = 0.05
solver.max_rounds_without_improvement = 5

result = solver.run_allocate()
```

**Bidding Process**:
1. **Bid Generation**: Each UAV calculates marginal utility for tasks
2. **Coalition Formation**: UAVs form coalitions for high-value tasks
3. **Auction Rounds**: Multiple rounds until convergence
4. **Winner Determination**: Highest value coalitions win tasks

**Bid Calculation**:
```python
def calculate_bid(self, uav, task, current_coalition):
    """Calculate marginal utility bid for UAV joining coalition."""
    
    # Base utility from resource contribution
    resource_utility = self._resource_contribution(uav, task)
    
    # Distance cost
    distance_cost = uav.position.distance_to(task.position) / uav.max_speed
    
    # Threat penalty
    threat_penalty = task.threat * uav.value
    
    # Marginal contribution
    marginal_utility = resource_utility - distance_cost - threat_penalty
    
    return max(0, marginal_utility)
```

**Best Use Cases**:
- Dynamic environments
- Heterogeneous UAV capabilities
- Market-based coordination
- Scalable to large swarms

---

### AuctionBiddingSolverAdvanced

**Description**: Advanced auction solver with sophisticated bidding strategies and coalition management.

**Additional Features**:
- Multi-round bidding
- Coalition stability analysis
- Adaptive bid strategies
- Performance optimization

---

## Distributed Solvers

### DistributedSolver

**Description**: Multi-agent distributed solver with message passing between UAVs.

**Architecture**:
- Each UAV runs local decision algorithm
- Message passing for coordination
- Consensus mechanisms
- Fault tolerance

**API Usage**:
```python
from solvers.distributed_solver import DistributedSolver

solver = DistributedSolver(uav_manager, task_manager, coalition_manager, hyper_params)

# Distributed parameters
hyper_params.communication_rounds = 10
hyper_params.consensus_threshold = 0.9
hyper_params.message_delay = 0.1  # seconds

result = solver.run_allocate()
```

**Message Types**:
```python
class Message:
    """Communication message between UAVs."""
    sender_id: int
    receiver_id: int
    message_type: str  # "proposal", "accept", "reject", "update"
    content: Dict
    timestamp: float
```

---

### DistributedSolver_MultiProcess

**Description**: True multi-process distributed solver with separate processes for each UAV.

**API Usage**:
```python
from solvers.distributed_multi_process_solver import DistributedSolver_MultiProcess

solver = DistributedSolver_MultiProcess(uav_manager, task_manager, coalition_manager, hyper_params)

# Multi-process parameters
hyper_params.process_timeout = 30.0
hyper_params.shared_memory = True
hyper_params.process_pool_size = 8

result = solver.run_allocate()
```

---

## Enumeration and Brute Force Solvers

### EnumerationSolver

**Description**: Exhaustive search solver for small-scale problems, providing optimal solutions.

**API Usage**:
```python
from solvers.enum_solver import EnumerationSolver

solver = EnumerationSolver(uav_manager, task_manager, coalition_manager, hyper_params)

# Enumeration parameters
hyper_params.max_combinations = 100000
hyper_params.pruning_enabled = True
hyper_params.early_termination = True

result = solver.run_allocate()
```

**Complexity**: O(2^(n×m)) where n=UAVs, m=tasks

**Best Use Cases**:
- Small problems (< 10 UAVs, < 5 tasks)
- Benchmarking other algorithms
- Optimal solution verification

---

### BruteForceSearchSolver

**Description**: Systematic search through solution space with pruning.

**Search Strategy**:
- Depth-first search
- Branch and bound pruning
- Constraint satisfaction

---

## Solver Comparison

| Solver | Scalability | Optimality | Real-time | Distributed | Complexity |
|--------|-------------|------------|-----------|-------------|------------|
| CSCI2024 | Medium | Near-optimal | Good | No | O(n²m) |
| IROS2024 | High | Good | Excellent | Yes | O(nm) |
| ICRA2024 | High | Good | Excellent | Yes | O(nm) |
| MILP | Low | Optimal | Poor | No | Exponential |
| NLP | Medium | Local optimal | Good | No | O(n³) |
| Auction | High | Good | Good | Yes | O(n²) |
| Distributed | Very High | Good | Excellent | Yes | O(n) |
| Enumeration | Very Low | Optimal | Poor | No | O(2^(nm)) |

## Performance Guidelines

### Small Scale (< 10 UAVs, < 5 tasks)
**Recommended**: EnumerationSolver, MILPSolver
```python
# Optimal solution for small problems
solver = EnumerationSolver(managers..., hyper_params)
```

### Medium Scale (10-50 UAVs, 5-20 tasks)
**Recommended**: ChinaScience2024_CoalitionFormationGame, NLPSolverScipy
```python
# Good balance of quality and speed
solver = ChinaScience2024_CoalitionFormationGame(managers..., hyper_params)
hyper_params.max_iter = 50
```

### Large Scale (50+ UAVs, 20+ tasks)
**Recommended**: IROS2024_CoalitionFormationGame, DistributedSolver, AuctionBiddingSolver
```python
# Scalable distributed solution
solver = IROS2024_CoalitionFormationGame(managers..., hyper_params)
hyper_params.max_iter = 30
```

### Real-time Applications
**Recommended**: IROS2024, ICRA2024, DistributedSolver
```python
# Fast convergence for real-time
solver = IROS2024_CoalitionFormationGame(managers..., hyper_params)
hyper_params.max_iter = 10
hyper_params.tolerance = 1e-2  # Relaxed tolerance
```

### Research and Benchmarking
**Recommended**: All solvers for comparison
```python
# Comprehensive comparison
solvers = [
    ChinaScience2024_CoalitionFormationGame,
    IROS2024_CoalitionFormationGame,
    AuctionBiddingSolver,
    MILPSolver,
]

results = {}
for solver_class in solvers:
    solver = solver_class(managers..., hyper_params)
    results[solver.type_name()] = solver.run_allocate()
```

### Parameter Tuning Guidelines

#### Coalition Formation Games
```python
# For resource-intensive tasks
hyper_params.alpha = 0.7  # Higher resource weight
hyper_params.beta = 0.2   # Lower distance weight
hyper_params.gamma = 0.1  # Lower threat weight

# For time-critical missions
hyper_params.alpha = 0.4
hyper_params.beta = 0.4   # Higher distance weight
hyper_params.gamma = 0.2
```

#### Auction Solvers
```python
# For stable coalitions
solver.bid_decay_factor = 0.98  # Slower decay
solver.min_bid_improvement = 0.02  # Lower threshold

# For dynamic environments
solver.bid_decay_factor = 0.90  # Faster decay
solver.min_bid_improvement = 0.10  # Higher threshold
```

This comprehensive solver documentation provides detailed information about each algorithm's implementation, usage patterns, and performance characteristics to help users select the most appropriate solver for their specific MRTA applications.