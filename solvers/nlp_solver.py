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

import numpy as np
from scipy.optimize import minimize


class NLPSolverScipy(MRTASolver):
    """
    Implements a NLP solver for task assignment using SciPy's optimizer.
    """

    def run_allocate(self):
        uavs = self.uav_manager.get_all()
        tasks = self.task_manager.get_all()
        num_uavs = len(uavs)
        num_tasks = len(tasks)

        # Mapping (i, j) to flat index in x
        def idx(i, j):
            return i * num_tasks + j

        # Initial guess
        x0 = np.zeros(num_uavs * num_tasks)

        # Objective function (negative because we minimize)
        def objective(x):
            task_completion = 0
            resource_util = 0
            distance_cost = 0
            threat = 0
            coalition_size = 0

            z = np.zeros(num_uavs)  # coalition indicator per UAV

            for i, uav in enumerate(uavs):
                assigned = 0
                for j, task in enumerate(tasks):
                    x_ij = x[idx(i, j)]
                    min_res = np.sum(np.minimum(uav.resources, task.required_resources))
                    task_completion += x_ij * min_res
                    resource_util += x_ij * min_res / (np.sum(uav.resources) + 1e-6)
                    distance_cost += x_ij * uav.position.distance_to(task.position)
                    threat += x_ij * task.threat
                    assigned = max(assigned, x_ij)
                z[i] = assigned
                coalition_size += z[i]

            # Weights (same as Pyomo version)
            return -(
                15.0 * task_completion
                + 5.0 * resource_util
                - 3.0 * distance_cost
                - 2.0 * threat
                - 1.0 * coalition_size
            )

        # Constraints

        constraints = []

        # Each UAV assigned to at most one task
        for i in range(num_uavs):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, i=i: 1.0 - sum(x[idx(i, j)] for j in range(num_tasks)),
                }
            )

        # Each task can receive up to len(uavs) UAVs (relaxed, or define your own logic)
        for j in range(num_tasks):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, j=j: len(uavs) - sum(x[idx(i, j)] for i in range(num_uavs)),
                }
            )

        # Variable bounds in [0, 1]
        bounds = [(0.0, 1.0) for _ in range(num_uavs * num_tasks)]

        # Solve with SLSQP (good for constrained continuous NLP)
        result = minimize(
            objective,
            x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"disp": True, "maxiter": 500},
        )

        # Extract solution and apply threshold
        allocation = {task.id: [] for task in tasks}
        x_opt = result.x

        for i, uav in enumerate(uavs):
            for j, task in enumerate(tasks):
                if x_opt[idx(i, j)] > 0.5:
                    allocation[task.id].append(uav.id)

        print(allocation)
        self.coalition_manager.update_from_assignment(allocation, uav_manager=self.uav_manager)


import pyomo.environ as pyo


class NLPSolverPyomo(MRTASolver):
    """
    Implements a NLP solver for task assignment with enhanced objective function.
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
        """Solve the task allocation problem using NLP with Pyomo."""
        model = pyo.ConcreteModel()

        # Get UAVs and tasks
        uavs = self.uav_manager.get_all()
        tasks = self.task_manager.get_all()
        uav_ids = [uav.id for uav in uavs]
        task_ids = [task.id for task in tasks]
        pairs = [(uav.id, task.id) for uav in uavs for task in tasks]

        # Continuous decision variables in [0, 1]
        model.x = pyo.Var(pairs, domain=pyo.UnitInterval)

        # Auxiliary variables for coalition size (1 if UAV is assigned to any task)
        model.z = pyo.Var(uav_ids, domain=pyo.UnitInterval)

        # Constraints: z[i] ≥ x[i, j] ∀j
        model.constraints = pyo.ConstraintList()
        for uav in uavs:
            for task in tasks:
                model.constraints.add(model.z[uav.id] >= model.x[(uav.id, task.id)])

        # Each UAV can be assigned to at most one task
        for uav in uavs:
            model.constraints.add(sum(model.x[(uav.id, task.id)] for task in tasks) <= 1)

        # Each task can be assigned multiple UAVs (or define limit if needed)
        for task in tasks:
            model.constraints.add(sum(model.x[(uav.id, task.id)] for uav in uavs) <= len(uavs))

        # Objective components
        def task_completion():
            return sum(
                model.x[(uav.id, task.id)]
                * np.sum(np.minimum(uav.resources, task.required_resources))
                for uav in uavs
                for task in tasks
            )

        def resource_utilization():
            return sum(
                model.x[(uav.id, task.id)]
                * np.sum(np.minimum(uav.resources, task.required_resources))
                / (np.sum(uav.resources) + 1e-6)
                for uav in uavs
                for task in tasks
            )

        def total_distance_cost():
            return sum(
                model.x[(uav.id, task.id)] * uav.position.distance_to(task.position)
                for uav in uavs
                for task in tasks
            )

        def threat_cost():
            return sum(model.x[(uav.id, task.id)] * task.threat for uav in uavs for task in tasks)

        def coalition_size():
            return sum(model.z[uav.id] for uav in uavs)

        # Weights (can be loaded from hyper_params)
        w_task_completion = 15.0
        w_resource_utilization = 5.0
        w_distance = 3.0
        w_threat = 2.0
        w_coalition = 1.0

        # Objective function
        model.obj = pyo.Objective(
            expr=(
                w_task_completion * task_completion()
                + w_resource_utilization * resource_utilization()
                - w_distance * total_distance_cost()
                - w_threat * threat_cost()
                - w_coalition * coalition_size()
            ),
            sense=pyo.maximize,
        )

        # Solve with an NLP solver (e.g., IPOPT)
        # conda install -c conda-forge ipopt
        solver = pyo.SolverFactory("ipopt")  # Make sure ipopt is installed
        solver.solve(model)

        # Extract the solution with a threshold
        allocation = {}
        for task in tasks:
            assigned_uavs = []
            for uav in uavs:
                val = pyo.value(model.x[(uav.id, task.id)])
                if val and val > 0.5:
                    assigned_uavs.append(uav.id)
            allocation[task.id] = assigned_uavs

        print(allocation)
        self.coalition_manager.update_from_assignment(allocation, uav_manager=self.uav_manager)
