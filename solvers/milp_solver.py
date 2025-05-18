from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import random
import numpy as np

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
from framework.utils import calculate_obtained_resources

from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpMaximize

log_level: LogLevel = LogLevel.SILENCE


class MILPSolver(MRTASolver):
    """
    Implements a MILP solver for task assignment.
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)

    @classmethod
    def type_name(cls):
        return "MILP"

    def run_allocate(self):
        """Solve the task allocation problem using MILP."""
        # Problem initialization
        prob = LpProblem("UAV_Task_Allocation", LpMaximize)

        # Get all UAVs and tasks
        uav_list = self.uav_manager.get_all()
        task_list = self.task_manager.get_all()

        # Create binary decision variables: x_ij = 1 if UAV i is assigned to task j
        x = LpVariable.dicts(
            "assignment",
            [(uav.id, task.id) for uav in uav_list for task in task_list],
            0,
            1,
            cat="Binary",
        )

        # Objective function components
        # 1. Maximize task completion (sum of resources allocated to tasks)
        task_completion = lpSum(
            x[(uav.id, task.id)] * np.sum(np.minimum(uav.resources, task.required_resources))
            for uav in uav_list
            for task in task_list
        )

        # 2. Maximize resource utilization (sum of resources used / resources available)
        resource_utilization = lpSum(
            x[(uav.id, task.id)]
            * np.sum(np.minimum(uav.resources, task.required_resources))
            / (np.sum(uav.resources) + 1e-6)  # Avoid division by zero
            for uav in uav_list
            for task in task_list
        )

        # 3. Minimize resource consumption (sum of resources allocated)
        resource_consumption = lpSum(
            x[(uav.id, task.id)] * np.sum(uav.resources) for uav in uav_list for task in task_list
        )

        # 4. Minimize total distance (sum of distances between UAVs and their assigned tasks)
        total_distance = lpSum(
            x[(uav.id, task.id)] * uav.position.distance_to(task.position)
            for uav in uav_list
            for task in task_list
        )

        # 5. Minimize threat cost (sum of threat indices of assigned tasks)
        threat_cost = lpSum(
            x[(uav.id, task.id)] * task.threat for uav in uav_list for task in task_list
        )

        # Combine objectives with weights (these could be part of hyper_params)
        # We maximize the positive terms and minimize the negative ones
        task_completion_weight = 15.0
        resource_utilization_weight = 5.0
        resource_consumption_weight = 5.0
        distance_weight = 5.0
        threat_weight = 2.0
        prob += (
            task_completion_weight * task_completion
            + resource_utilization_weight * resource_utilization
            - resource_consumption_weight * resource_consumption
            - distance_weight * total_distance
            - threat_weight * threat_cost
        )

        # Constraints
        # 1. Each UAV can be assigned to at most one task
        for uav in uav_list:
            prob += lpSum(x[(uav.id, task.id)] for task in task_list) <= 1

        # Solve the problem
        prob.solve()

        # Extract the solution
        allocation = {}
        for task in task_list:
            assigned_uavs = []
            for uav in uav_list:
                if x[(uav.id, task.id)].varValue == 1:
                    assigned_uavs.append(uav.id)
            allocation[task.id] = assigned_uavs
        print(allocation)
        self.coalition_manager.update_from_assignment(allocation, uav_manager=self.uav_manager)


import pyomo.environ as pyo


class MILPSolverPyomo(MRTASolver):
    """
    Implements a MILP solver for task assignment using Pyomo.
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)

    def run_allocate(self):
        """Solve the task allocation problem using MILP with Pyomo."""
        model = pyo.ConcreteModel()

        # Get UAVs and tasks
        uav_list = self.uav_manager.get_all()
        task_list = self.task_manager.get_all()

        pairs = [(uav.id, task.id) for uav in uav_list for task in task_list]

        # Decision variables: binary assignment vars
        model.x = pyo.Var(pairs, domain=pyo.Binary)

        # Objective components
        def task_completion():
            return sum(
                model.x[(uav.id, task.id)]
                * np.sum(np.minimum(uav.resources, task.required_resources))
                for uav in uav_list
                for task in task_list
            )

        def resource_utilization():
            return sum(
                model.x[(uav.id, task.id)]
                * np.sum(np.minimum(uav.resources, task.required_resources))
                / (np.sum(uav.resources) + 1e-6)
                for uav in uav_list
                for task in task_list
            )

        def resource_consumption():
            return sum(
                model.x[(uav.id, task.id)] * np.sum(uav.resources)
                for uav in uav_list
                for task in task_list
            )

        def total_distance():
            return sum(
                model.x[(uav.id, task.id)] * uav.position.distance_to(task.position)
                for uav in uav_list
                for task in task_list
            )

        def threat_cost():
            return sum(
                model.x[(uav.id, task.id)] * task.threat for uav in uav_list for task in task_list
            )

        # Weights from hyper-parameters or constants
        task_completion_weight = 15.0
        resource_utilization_weight = 5.0
        resource_consumption_weight = 3.0
        distance_weight = 3.0
        threat_weight = 2.0

        # Combined objective
        model.objective = pyo.Objective(
            expr=(
                task_completion_weight * task_completion()
                + resource_utilization_weight * resource_utilization()
                - resource_consumption_weight * resource_consumption()
                - distance_weight * total_distance()
                - threat_weight * threat_cost()
            ),
            sense=pyo.maximize,
        )

        # Constraints
        model.constraints = pyo.ConstraintList()

        # Each UAV can be assigned to at most one task
        for uav in uav_list:
            model.constraints.add(sum(model.x[(uav.id, task.id)] for task in task_list) <= 1)

        # Solve the model
        solver = pyo.SolverFactory("glpk")  # sudo apt install glpk-utils
        solver.solve(model)

        # Extract solution
        allocation = {}
        for task in task_list:
            assigned_uavs = []
            for uav in uav_list:
                if pyo.value(model.x[(uav.id, task.id)]) == 1:
                    assigned_uavs.append(uav.id)
            allocation[task.id] = assigned_uavs

        # print(allocation)
        self.coalition_manager.update_from_assignment(allocation, uav_manager=self.uav_manager)
