from typing import List, Dict, Optional, Type
from dataclasses import dataclass, field
import random
import numpy as np
import time

from .base import HyperParams, LogLevel
from .uav import UAV, UAVManager
from .task import Task, TaskManager
from .coalition_manager import CoalitionManager
from .mrta_solver import MRTASolver


@dataclass
class DynamicEnvironmentConfig:
    """Configuration for the dynamic environment simulation."""
    # Task generation parameters
    task_generation_probability: float = 0.1  # Probability of generating a new task each time step
    task_completion_probability: float = 0.05  # Probability of completing a task each time step
    task_update_probability: float = 0.1  # Probability of updating a task's requirements each time step
    
    # UAV parameters
    uav_damage_probability: float = 0.02  # Probability of a UAV getting damaged each time step
    uav_repair_probability: float = 0.1  # Probability of a damaged UAV getting repaired each time step
    uav_update_probability: float = 0.05  # Probability of updating a UAV's capabilities each time step
    
    # Task generation ranges
    task_required_resources_range: tuple = (1, 5)  # Range for task resource requirements
    task_threat_range: tuple = (0.1, 0.9)  # Range for task threat levels
    task_time_window_range: tuple = (5, 20)  # Range for task time windows
    
    # UAV update ranges
    uav_resource_update_range: tuple = (-1, 1)  # Range for UAV resource updates
    uav_value_update_range: tuple = (-0.5, 0.5)  # Range for UAV value updates
    uav_speed_update_range: tuple = (-0.2, 0.2)  # Range for UAV speed updates

"""
run_simulation:
for step in range(num_steps):
    self.update_env()
    self.update_allocate()
    
    self.print_state()

"""


class DynamicMRTASolver(MRTASolver):
    """Base class for dynamic Multi-Robot Task Assignment solvers."""
    
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        env_config: Optional[DynamicEnvironmentConfig] = None,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        self.env_config = env_config or DynamicEnvironmentConfig()
        self.time_step = 0
        self.damaged_uavs: Dict[int, float] = {}  # UAV ID -> damage level (0-1)
        self.task_completion_times: Dict[int, float] = {}  # Task ID -> completion time
        
    def update_environment(self):
        """Update the environment state for the current time step."""
        self.time_step += 1
        
        # Update tasks
        self._update_tasks()
        
        # Update UAVs
        self._update_uavs()
        
        # Update coalitions
        self._update_coalitions()
        
    def _update_tasks(self):
        """Update task states (generate, complete, update)."""
        # Generate new tasks
        if random.random() < self.env_config.task_generation_probability:
            self._generate_new_task()
            
        # Complete existing tasks
        for task in self.task_manager.get_all():
            if task.id == TaskManager.free_uav_task_id:
                continue
                
            if random.random() < self.env_config.task_completion_probability:
                self._complete_task(task)
                
        # Update task requirements
        for task in self.task_manager.get_all():
            if task.id == TaskManager.free_uav_task_id:
                continue
                
            if random.random() < self.env_config.task_update_probability:
                self._update_task_requirements(task)
                
    def _update_uavs(self):
        """Update UAV states (damage, repair, update)."""
        # Damage UAVs
        for uav in self.uav_manager.get_all():
            if random.random() < self.env_config.uav_damage_probability:
                self._damage_uav(uav)
                
        # Repair damaged UAVs
        for uav_id in list(self.damaged_uavs.keys()):
            if random.random() < self.env_config.uav_repair_probability:
                self._repair_uav(uav_id)
                
        # Update UAV capabilities
        for uav in self.uav_manager.get_all():
            if random.random() < self.env_config.uav_update_probability:
                self._update_uav_capabilities(uav)
                
    def _update_coalitions(self):
        """Update coalition assignments based on current state."""
        # Remove completed tasks from coalitions
        completed_task_ids = list(self.task_completion_times.keys())
        for task_id in completed_task_ids:
            if task_id in self.coalition_manager.task2coalition:
                coalition = self.coalition_manager.get_coalition(task_id)
                for uav_id in coalition:
                    self.coalition_manager.unassign(uav_id)
                    
        # Remove damaged UAVs from coalitions
        for uav_id in self.damaged_uavs:
            if uav_id in self.coalition_manager.uavid2taskid:
                self.coalition_manager.unassign(uav_id)
                
    def _generate_new_task(self):
        """Generate a new task with random requirements."""
        task_id = max(self.task_manager.get_ids()) + 1
        position = [
            random.uniform(0, self.hyper_params.map_shape[i])
            for i in range(3)
        ]
        required_resources = [
            random.randint(*self.env_config.task_required_resources_range)
            for _ in range(self.hyper_params.resources_num)
        ]
        time_window = [
            self.time_step,
            self.time_step + random.randint(*self.env_config.task_time_window_range)
        ]
        threat = random.uniform(*self.env_config.task_threat_range)
        
        new_task = Task(
            id=task_id,
            position=position,
            required_resources=required_resources,
            time_window=time_window,
            threat=threat
        )
        
        self.task_manager.add(new_task)
        
    def _complete_task(self, task: Task):
        """Mark a task as completed."""
        if task.id not in self.task_completion_times:
            self.task_completion_times[task.id] = self.time_step
            
    def _update_task_requirements(self, task: Task):
        """Update a task's resource requirements."""
        # Randomly modify resource requirements
        for i in range(len(task.required_resources)):
            task.required_resources[i] = max(1, task.required_resources[i] + 
                random.randint(-1, 1))
                
    def _damage_uav(self, uav: UAV):
        """Damage a UAV, reducing its capabilities."""
        damage_level = random.uniform(0.2, 0.8)  # 20-80% damage
        self.damaged_uavs[uav.id] = damage_level
        
        # Reduce UAV capabilities based on damage
        uav.resources *= (1 - damage_level)
        uav.value *= (1 - damage_level)
        uav.max_speed *= (1 - damage_level)
        
    def _repair_uav(self, uav_id: int):
        """Repair a damaged UAV."""
        if uav_id in self.damaged_uavs:
            uav = self.uav_manager.get(uav_id)
            repair_factor = 1 + random.uniform(0, 0.2)  # 0-20% repair bonus
            
            # Restore UAV capabilities
            uav.resources *= repair_factor
            uav.value *= repair_factor
            uav.max_speed *= repair_factor
            
            del self.damaged_uavs[uav_id]
            
    def _update_uav_capabilities(self, uav: UAV):
        """Update a UAV's capabilities."""
        # Update resources
        for i in range(len(uav.resources)):
            uav.resources[i] = max(1, uav.resources[i] + 
                random.uniform(*self.env_config.uav_resource_update_range))
                
        # Update value
        uav.value = max(1, uav.value + 
            random.uniform(*self.env_config.uav_value_update_range))
            
        # Update speed
        uav.max_speed = max(1, uav.max_speed + 
            random.uniform(*self.env_config.uav_speed_update_range))
            
    def run_simulation(self, num_steps: int = 100):
        """Run the dynamic task assignment simulation."""
        for step in range(num_steps):
            print(f"\nTime step {step + 1}/{num_steps}")
            
            # Update environment
            self.update_environment()
            
            # Run task assignment
            self.run_allocate()
            
            # Print current state
            self._print_state()
            
    def _print_state(self):
        """Print the current state of the simulation."""
        print("\nCurrent State:")
        print(f"Active Tasks: {len(self.task_manager) - 1}")  # Exclude free UAV task
        print(f"Damaged UAVs: {len(self.damaged_uavs)}")
        print(f"Completed Tasks: {len(self.task_completion_times)}")
        
    @classmethod
    def type_name(cls):
        return "DynamicMRTA" 

if __name__ == "__main__":
    uav_manager = UAVManager()
    task_manager = TaskManager()
    coalition_manager = CoalitionManager()
    hyper_params = HyperParams()
    env_config = DynamicEnvironmentConfig()
    solver = DynamicMRTASolver(uav_manager, task_manager, coalition_manager, hyper_params, env_config)
    solver.run_simulation(num_steps=100)