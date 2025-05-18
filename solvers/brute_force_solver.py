from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
import time
import numpy as np
from tqdm import tqdm

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
from framework.utils import calculate_obtained_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BruteForceSolver")

@dataclass
class SearchState:
    """Represents the current state during brute force search."""
    assignment: Dict[int, List[int]] = field(default_factory=dict)
    unassigned_uavs: Set[int] = field(default_factory=set)
    current_score: float = 0.0
    best_possible_score: float = float('inf')
    depth: int = 0

class BruteForceSearchSolver(MRTASolver):
    """
    An improved brute force solver for UAV task assignment that supports multiple UAVs per task
    and uses efficient search strategies to find optimal allocations.
    
    Features:
    - Multiple UAV assignments per task
    - Efficient search with pruning
    - Progress tracking and logging
    - Configurable search parameters
    - Early termination on optimal solution
    """
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        max_coalition_size: int = 5,
        time_limit: float = 300.0,  # 5 minutes default
        early_termination_threshold: float = 0.95,  # 95% of theoretical maximum
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        self.max_coalition_size = max_coalition_size
        self.time_limit = time_limit
        self.early_termination_threshold = early_termination_threshold
        self.start_time = 0.0
        self.nodes_explored = 0
        self.pruned_nodes = 0
        
    @classmethod
    def type_name(cls):
        return "BruteForceSearch"
    
    def run_allocate(self) -> Dict[int, List[int]]:
        """
        Run the brute force search algorithm to find optimal task assignments.
        
        Returns:
            Dict[int, List[int]]: The best found assignment mapping task IDs to lists of UAV IDs
        """
        logger.info("Starting brute force search for optimal task assignment")
        self.start_time = time.time()
        self.nodes_explored = 0
        self.pruned_nodes = 0
        
        # Initialize search state
        initial_state = SearchState(
            unassigned_uavs=set(self.uav_manager.get_ids()),
            assignment={task.id: [] for task in self.task_manager.get_all() 
                       if task.id != self.task_manager.free_uav_task_id}
        )
        
        # Calculate theoretical maximum score for early termination
        theoretical_max = self._calculate_theoretical_max_score()
        logger.info(f"Theoretical maximum score: {theoretical_max:.2f}")
        
        # Start search with progress bar
        best_assignment = None
        best_score = float('-inf')
        
        with tqdm(total=len(self.uav_manager), desc="Searching assignments") as pbar:
            try:
                best_assignment, best_score = self._search(
                    initial_state,
                    best_assignment,
                    best_score,
                    theoretical_max,
                    pbar
                )
            except TimeoutError:
                logger.warning("Search terminated due to time limit")
        
        # Log final results
        logger.info(f"Search completed in {time.time() - self.start_time:.2f} seconds")
        logger.info(f"Nodes explored: {self.nodes_explored}")
        logger.info(f"Nodes pruned: {self.pruned_nodes}")
        logger.info(f"Best score found: {best_score:.2f}")
        
        # Update coalition manager with best assignment
        if best_assignment:
            self.coalition_manager.update_from_assignment(best_assignment)
        
        return best_assignment
    
    def _search(
        self,
        state: SearchState,
        best_assignment: Optional[Dict[int, List[int]]],
        best_score: float,
        theoretical_max: float,
        pbar: tqdm
    ) -> Tuple[Optional[Dict[int, List[int]]], float]:
        """
        Recursive search function with pruning and early termination.
        
        Args:
            state: Current search state
            best_assignment: Best assignment found so far
            best_score: Score of best assignment found so far
            theoretical_max: Theoretical maximum possible score
            pbar: Progress bar for tracking
            
        Returns:
            Tuple of (best assignment, best score)
        """
        # Check time limit
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError("Search time limit exceeded")
        
        # Check early termination
        if best_score >= theoretical_max * self.early_termination_threshold:
            return best_assignment, best_score
        
        self.nodes_explored += 1
        
        # If all UAVs are assigned, evaluate the assignment
        if not state.unassigned_uavs:
            score = self._evaluate_assignment(state.assignment)
            if score > best_score:
                best_score = score
                best_assignment = state.assignment.copy()
                pbar.set_postfix({"best_score": f"{best_score:.2f}"})
            return best_assignment, best_score
        
        # Get next UAV to assign
        uav_id = state.unassigned_uavs.pop()
        
        # Try assigning UAV to each task
        for task_id in state.assignment.keys():
            # Check coalition size limit
            if len(state.assignment[task_id]) >= self.max_coalition_size:
                continue
                
            # Try adding UAV to task
            state.assignment[task_id].append(uav_id)
            
            # Calculate potential score improvement
            potential_score = self._calculate_potential_score(state)
            
            # Prune if potential score is worse than current best
            if potential_score <= best_score:
                self.pruned_nodes += 1
                state.assignment[task_id].pop()
                continue
            
            # Recursive search
            best_assignment, best_score = self._search(
                state, best_assignment, best_score, theoretical_max, pbar
            )
            
            # Backtrack
            state.assignment[task_id].pop()
        
        # Try leaving UAV unassigned
        best_assignment, best_score = self._search(
            state, best_assignment, best_score, theoretical_max, pbar
        )
        
        # Restore state
        state.unassigned_uavs.add(uav_id)
        pbar.update(1)
        
        return best_assignment, best_score
    
    def _evaluate_assignment(self, assignment: Dict[int, List[int]]) -> float:
        """
        Evaluate the quality of a task assignment.
        
        Args:
            assignment: Dictionary mapping task IDs to lists of assigned UAV IDs
            
        Returns:
            float: Score of the assignment
        """
        total_score = 0.0
        
        for task_id, uav_ids in assignment.items():
            if not uav_ids:  # Skip empty assignments
                continue
                
            task = self.task_manager.get(task_id)
            coalition = uav_ids
            
            # Calculate obtained resources
            obtained_resources = calculate_obtained_resources(
                coalition, self.uav_manager, self.hyper_params.resources_num
            )
            
            # Calculate resource satisfaction
            resource_satisfaction = np.sum(obtained_resources >= task.required_resources) / self.hyper_params.resources_num
            
            # Calculate resource efficiency
            if np.sum(obtained_resources) > 0:
                resource_efficiency = 1 - np.sum(np.maximum(obtained_resources - task.required_resources, 0)) / np.sum(obtained_resources)
            else:
                resource_efficiency = 0
                
            # Calculate coalition utility
            coalition_utility = 0
            for uav_id in uav_ids:
                uav = self.uav_manager.get(uav_id)
                benefit = self._calculate_uav_benefit(uav, task, coalition, obtained_resources)
                coalition_utility += benefit
            
            # Combine factors with weights
            weights = {
                "resource_satisfaction": 3.0,
                "resource_efficiency": 2.0,
                "coalition_utility": 1.0
            }
            
            task_score = (
                weights["resource_satisfaction"] * resource_satisfaction +
                weights["resource_efficiency"] * resource_efficiency +
                weights["coalition_utility"] * coalition_utility
            )
            
            total_score += task_score
            
        return total_score
    
    def _calculate_uav_benefit(
        self,
        uav: UAV,
        task: Task,
        coalition: List[int],
        obtained_resources: np.ndarray
    ) -> float:
        """Calculate the benefit of a UAV joining a task coalition."""
        # Resource contribution
        resource_contribution = np.sum(np.minimum(uav.resources, task.required_resources))
        
        # Path cost
        distance = uav.position.distance_to(task.position)
        max_distance = np.linalg.norm(self.hyper_params.map_shape)
        path_cost = 1 - (distance / max_distance)
        
        # Threat cost
        threat_cost = uav.value * task.threat
        
        # Time window compatibility
        arrival_time = distance / uav.max_speed
        time_compatibility = 1.0 if arrival_time <= task.time_window[1] else 0.0
        
        # Combine factors
        weights = {
            "resource": 3.0,
            "path": 2.0,
            "threat": 1.0,
            "time": 4.0
        }
        
        benefit = (
            weights["resource"] * resource_contribution +
            weights["path"] * path_cost -
            weights["threat"] * threat_cost +
            weights["time"] * time_compatibility
        )
        
        return max(0, benefit)
    
    def _calculate_potential_score(self, state: SearchState) -> float:
        """Calculate the potential maximum score for the current state."""
        current_score = self._evaluate_assignment(state.assignment)
        
        # Estimate maximum possible improvement from remaining UAVs
        max_improvement = 0
        for uav_id in state.unassigned_uavs:
            uav = self.uav_manager.get(uav_id)
            max_uav_benefit = 0
            
            for task_id in state.assignment.keys():
                task = self.task_manager.get(task_id)
                if len(state.assignment[task_id]) < self.max_coalition_size:
                    # Calculate maximum possible benefit
                    max_benefit = self._calculate_max_possible_benefit(uav, task)
                    max_uav_benefit = max(max_uav_benefit, max_benefit)
            
            max_improvement += max_uav_benefit
        
        return current_score + max_improvement
    
    def _calculate_max_possible_benefit(self, uav: UAV, task: Task) -> float:
        """Calculate the maximum possible benefit a UAV could contribute to a task."""
        # Maximum resource contribution
        max_resource = np.sum(np.minimum(uav.resources, task.required_resources))
        
        # Best case path cost (assuming minimum distance)
        min_path_cost = 1.0
        
        # Best case time compatibility
        best_time_compatibility = 1.0
        
        # Combine factors with weights
        weights = {
            "resource": 3.0,
            "path": 2.0,
            "time": 4.0
        }
        
        return (
            weights["resource"] * max_resource +
            weights["path"] * min_path_cost +
            weights["time"] * best_time_compatibility
        )
    
    def _calculate_theoretical_max_score(self) -> float:
        """Calculate the theoretical maximum possible score."""
        max_score = 0
        
        for task in self.task_manager.get_all():
            if task.id == self.task_manager.free_uav_task_id:
                continue
                
            # Calculate maximum possible score for each task
            task_max_score = 0
            for uav in self.uav_manager.get_all():
                max_benefit = self._calculate_max_possible_benefit(uav, task)
                task_max_score += max_benefit
            
            max_score += task_max_score
        
        return max_score 