from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
import random
import numpy as np

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
from framework.utils import calculate_obtained_resources

log_level: LogLevel = LogLevel.SILENCE


from . import iros2024


class AuctionBiddingSolverKimi(MRTASolver):
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)

    def run_allocate(self):
        """
        拍卖竞标算法进行任务分配
        """
        # 初始化每个任务的当前出价和中标无人机
        task_bids = {task.id: {"bid": 0, "winner": None} for task in self.task_manager.get_all()}

        # 迭代次数
        iter_cnt = 0
        while True:
            # print(f"iter {iter_cnt}")
            if iter_cnt > self.hyper_params.max_iter:
                # print(f"reach max iter {self.hyper_params.max_iter}")
                break
            iter_cnt += 1

            # 遍历每个无人机，进行出价
            for uav in self.uav_manager.get_all():
                # 计算该无人机对每个任务的估价
                task_utilities = {}
                for task in self.task_manager.get_all():
                    utility = self.calculate_uav_task_utility(uav, task)
                    task_utilities[task.id] = utility

                # 选择估价最高的任务进行出价
                target_task_id = max(task_utilities, key=task_utilities.get)
                target_task_bid = task_utilities[target_task_id]

                # 如果出价高于当前最高出价，则更新中标无人机
                if target_task_bid > task_bids[target_task_id]["bid"]:
                    task_bids[target_task_id]["bid"] = target_task_bid
                    task_bids[target_task_id]["winner"] = uav.id

            # 检查是否达到稳定状态
            if self.check_stability(task_bids):
                break

        # 根据中标结果分配任务
        for task_id, bid_info in task_bids.items():
            if bid_info["winner"] is not None:
                self.coalition_manager.assign(bid_info["winner"], task_id)

        return task_bids

    def calculate_uav_task_utility(self, uav: UAV, task: Task) -> float:
        """
        计算无人机对任务的估价
        """
        # 飞行成本
        fly_energy = uav.cal_fly_energy(task.position)
        max_fly_energy = max(uav.cal_fly_energy(t.position) for t in self.task_manager.get_all())
        min_fly_energy = min(uav.cal_fly_energy(t.position) for t in self.task_manager.get_all())
        norm_fly_energy_cost = iros2024.min_max_norm(fly_energy, min_fly_energy, max_fly_energy)

        # 悬停成本
        hover_energy = uav.cal_hover_energy(task.execution_time)
        max_hover_energy = max(
            uav.cal_hover_energy(t.execution_time) for t in self.task_manager.get_all()
        )
        min_hover_energy = min(
            uav.cal_hover_energy(t.execution_time) for t in self.task_manager.get_all()
        )
        norm_hover_energy_cost = iros2024.min_max_norm(
            hover_energy, min_hover_energy, max_hover_energy
        )

        # 合作成本
        cooperation_cost = (
            len(self.coalition_manager.get_coalition(task.id)) ** 2 / self.uav_manager.size()
        )

        # 成本
        cost = norm_fly_energy_cost + norm_hover_energy_cost + cooperation_cost

        # 任务满足率
        obtained_resources = calculate_obtained_resources(
            [uav.id], self.uav_manager, self.hyper_params.resources_num
        )
        task_satisfaction_rate = (
            np.sum(obtained_resources >= task.required_resources) / self.hyper_params.resources_num
        )
        task_satisfaction_rate = min(task_satisfaction_rate, 1)

        # 资源利用率
        if np.sum(obtained_resources) == 0:
            raise ValueError("obtained_resources should not be 0")
        else:
            task_resource_waste_rate = np.sum(
                np.maximum(obtained_resources - task.required_resources, 0)
            ) / np.sum(obtained_resources)
        task_resource_use_rate = 1 - task_resource_waste_rate

        # 估价
        utility = (task_satisfaction_rate + task_resource_use_rate) / cost
        return utility

    def check_stability(self, task_bids: Dict) -> bool:
        """
        检查是否达到稳定状态
        """
        # 如果所有任务的中标无人机没有变化，则认为达到稳定状态
        for task_id, bid_info in task_bids.items():
            if bid_info["winner"] is None:
                return False
            current_winner = bid_info["winner"]
            current_bid = bid_info["bid"]
            for uav in self.uav_manager.get_all():
                if uav.id == current_winner:
                    continue
                utility = self.calculate_uav_task_utility(uav, self.task_manager.get(task_id))
                if utility > current_bid:
                    return False
        return True


import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AuctionSolver")


@dataclass
class Bid:
    """Class to represent a bid from a UAV for a task."""

    uav_id: int
    task_id: int
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Coalition:
    """Class to represent a coalition of UAVs for a task."""

    task_id: int
    uav_ids: Set[int] = field(default_factory=set)
    total_value: float = 0.0
    last_updated: float = field(default_factory=time.time)


class AuctionBiddingSolver(MRTASolver):
    """
    Enhanced Auction-based Task Allocation Solver

    This solver uses a combinatorial auction mechanism to allocate tasks to UAVs,
    allowing multiple UAVs to form coalitions for tasks that require more resources.

    Features:
    - Supports multiple UAVs per task (coalition formation)
    - Uses marginal utility for bid calculation
    - Implements bid decay over time to prevent stagnation
    - Provides detailed metrics and logging
    - Handles resource constraints and task requirements
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        self.max_coalition_size = 5  # Maximum number of UAVs in a coalition
        self.bid_decay_factor = 0.95  # Decay factor for bids over iterations
        self.min_bid_improvement = 0.05  # Minimum relative improvement for a bid to be considered
        self.max_rounds_without_improvement = (
            5  # Maximum rounds without improvement before termination
        )

    @classmethod
    def type_name(cls):
        return "AuctionBidding"

    def run_allocate(self):
        """
        Run the auction-based task allocation algorithm.

        This method implements a combinatorial auction where UAVs bid on tasks
        and can form coalitions to satisfy task requirements.
        """
        # Initialize coalitions for each task
        coalitions = {
            task.id: Coalition(task_id=task.id)
            for task in self.task_manager.get_all()
            if task.id != self.task_manager.free_uav_task_id
        }

        # Track best bids for each UAV
        uav_best_bids = {uav.id: None for uav in self.uav_manager.get_all()}

        # Track metrics for termination condition
        rounds_without_improvement = 0
        previous_total_utility = 0

        # Main auction loop
        for iteration in range(self.hyper_params.max_iter):
            logger.debug(f"Starting auction iteration {iteration}")

            # Track if any changes were made in this round
            changes_made = False
            total_utility = 0

            # Each UAV submits bids
            for uav in self.uav_manager.get_all():
                # Calculate bids for each task
                bids = self._calculate_bids_for_uav(uav, coalitions)

                if not bids:
                    continue

                # Find the best bid
                best_bid = max(bids, key=lambda b: b.value)

                # Only consider the bid if it's significantly better than current assignment
                current_task_id = self._get_current_task_for_uav(uav.id, coalitions)

                if self._should_reassign_uav(uav.id, current_task_id, best_bid, uav_best_bids):
                    # Remove UAV from current coalition if assigned
                    if current_task_id is not None:
                        self._remove_uav_from_coalition(uav.id, current_task_id, coalitions)

                    # Add UAV to new coalition
                    self._add_uav_to_coalition(best_bid, coalitions)

                    # Update UAV's best bid
                    uav_best_bids[uav.id] = best_bid
                    changes_made = True

            # Calculate total utility of current allocation
            total_utility = self._calculate_total_utility(coalitions)

            # Update termination metrics
            if (
                abs(total_utility - previous_total_utility)
                < self.min_bid_improvement * previous_total_utility
            ):
                rounds_without_improvement += 1
            else:
                rounds_without_improvement = 0

            previous_total_utility = total_utility

            # Log progress
            logger.info(
                f"Iteration {iteration}: Total utility = {total_utility:.2f}, "
                + f"Changes made = {changes_made}, "
                + f"Rounds without improvement = {rounds_without_improvement}"
            )

            # Check termination conditions
            if (
                not changes_made
                or rounds_without_improvement >= self.max_rounds_without_improvement
            ):
                logger.info(f"Auction converged after {iteration + 1} iterations")
                break

        # Apply final allocation to coalition manager
        self._apply_allocation_to_coalition_manager(coalitions)

        return coalitions

    def _calculate_bids_for_uav(self, uav: UAV, coalitions: Dict[int, Coalition]) -> List[Bid]:
        """Calculate bids for all tasks for a given UAV based on marginal utility."""
        bids = []

        for task in self.task_manager.get_all():
            # Skip the free UAV task
            if task.id == self.task_manager.free_uav_task_id:
                continue

            # Skip if coalition is already at maximum size
            if len(coalitions[task.id].uav_ids) >= self.max_coalition_size:
                continue

            # Calculate marginal utility
            utility_before = self._calculate_coalition_utility(task, coalitions[task.id].uav_ids)

            # Create temporary coalition with this UAV added
            temp_coalition = coalitions[task.id].uav_ids.copy()
            temp_coalition.add(uav.id)

            utility_after = self._calculate_coalition_utility(task, temp_coalition)

            # Marginal utility is the improvement this UAV brings
            marginal_utility = utility_after - utility_before

            # Only create a bid if the UAV improves the coalition
            if marginal_utility > 0:
                bids.append(Bid(uav_id=uav.id, task_id=task.id, value=marginal_utility))

        return bids

    def _calculate_coalition_utility(self, task: Task, uav_ids: Set[int]) -> float:
        """Calculate the utility of a coalition for a given task."""
        if not uav_ids:
            return 0.0

        # Convert set to list for compatibility with other functions
        uav_id_list = list(uav_ids)

        # Calculate obtained resources
        obtained_resources = calculate_obtained_resources(
            uav_id_list, self.uav_manager, self.hyper_params.resources_num
        )

        # Calculate resource satisfaction
        resource_satisfaction = (
            np.sum(obtained_resources >= task.required_resources) / self.hyper_params.resources_num
        )

        # Calculate resource efficiency (penalize over-allocation)
        if np.sum(obtained_resources) == 0:
            resource_efficiency = 0
        else:
            over_allocation = np.sum(np.maximum(obtained_resources - task.required_resources, 0))
            resource_efficiency = 1 - (over_allocation / np.sum(obtained_resources))
            resource_efficiency = max(0, resource_efficiency)  # Ensure non-negative

        # Calculate distance cost
        uavs = [self.uav_manager.get(uav_id) for uav_id in uav_ids]
        avg_distance = np.mean([uav.position.distance_to(task.position) for uav in uavs])
        max_distance = np.sqrt(sum(d**2 for d in self.hyper_params.map_shape))
        normalized_distance = 1 - (avg_distance / max_distance)

        # Calculate time window compatibility
        avg_arrival_time = np.mean(
            [
                self.uav_manager.get(uav_id).position.distance_to(task.position)
                / self.uav_manager.get(uav_id).max_speed
                for uav_id in uav_ids
            ]
        )
        time_window_compatibility = 1.0 if avg_arrival_time <= task.time_window[1] else 0.0

        # Calculate threat cost
        avg_value = np.mean([self.uav_manager.get(uav_id).value for uav_id in uav_ids])
        normalized_threat = 1 - task.threat * avg_value / 10.0  # Assuming max threat * value = 10

        # Calculate coalition size efficiency (prefer smaller coalitions)
        size_efficiency = 1.0 / len(uav_ids) if len(uav_ids) > 0 else 0

        # Combine factors with weights
        weights = {
            "resource_satisfaction": 3.0,
            "resource_efficiency": 2.0,
            "distance": 1.5,
            "time_window": 4.0,
            "threat": 1.0,
            "size_efficiency": 0.5,
        }

        utility = (
            weights["resource_satisfaction"] * resource_satisfaction
            + weights["resource_efficiency"] * resource_efficiency
            + weights["distance"] * normalized_distance
            + weights["time_window"] * time_window_compatibility
            + weights["threat"] * normalized_threat
            + weights["size_efficiency"] * size_efficiency
        )

        return utility

    def _get_current_task_for_uav(self, uav_id: int, coalitions: Dict[int, Coalition]) -> int:
        """Find the current task assignment for a UAV."""
        for task_id, coalition in coalitions.items():
            if uav_id in coalition.uav_ids:
                return task_id
        return None

    def _should_reassign_uav(
        self, uav_id: int, current_task_id: int, new_bid: Bid, uav_best_bids: Dict[int, Bid]
    ) -> bool:
        """Determine if a UAV should be reassigned based on the new bid."""
        # If UAV is not currently assigned, accept any positive bid
        if current_task_id is None:
            return new_bid.value > 0

        # If UAV is bidding for the same task, only accept if significantly better
        if current_task_id == new_bid.task_id:
            return False

        # Get previous best bid for this UAV
        previous_bid = uav_best_bids[uav_id]

        # If no previous bid, accept the new one
        if previous_bid is None:
            return True

        # Accept if new bid is significantly better than previous
        return new_bid.value > previous_bid.value * (1 + self.min_bid_improvement)

    def _remove_uav_from_coalition(
        self, uav_id: int, task_id: int, coalitions: Dict[int, Coalition]
    ) -> None:
        """Remove a UAV from a coalition."""
        if task_id in coalitions and uav_id in coalitions[task_id].uav_ids:
            coalitions[task_id].uav_ids.remove(uav_id)
            coalitions[task_id].last_updated = time.time()

            # Recalculate coalition value
            coalition_value = self._calculate_coalition_utility(
                self.task_manager.get(task_id), coalitions[task_id].uav_ids
            )
            coalitions[task_id].total_value = coalition_value

    def _add_uav_to_coalition(self, bid: Bid, coalitions: Dict[int, Coalition]) -> None:
        """Add a UAV to a coalition based on a bid."""
        coalitions[bid.task_id].uav_ids.add(bid.uav_id)
        coalitions[bid.task_id].last_updated = time.time()

        # Recalculate coalition value
        coalition_value = self._calculate_coalition_utility(
            self.task_manager.get(bid.task_id), coalitions[bid.task_id].uav_ids
        )
        coalitions[bid.task_id].total_value = coalition_value

    def _calculate_total_utility(self, coalitions: Dict[int, Coalition]) -> float:
        """Calculate the total utility of all coalitions."""
        return sum(coalition.total_value for coalition in coalitions.values())

    def _apply_allocation_to_coalition_manager(self, coalitions: Dict[int, Coalition]) -> None:
        """Apply the final allocation to the coalition manager."""
        # First, ensure all UAVs are unassigned
        for uav in self.uav_manager.get_all():
            self.coalition_manager.unassign(uav.id)

        # Then, assign UAVs to their tasks based on coalitions
        for task_id, coalition in coalitions.items():
            for uav_id in coalition.uav_ids:
                self.coalition_manager.assign(uav_id, task_id)

        logger.info("Final allocation applied to coalition manager")

        # Log statistics about the allocation
        assigned_uavs = sum(len(coalition.uav_ids) for coalition in coalitions.values())
        total_uavs = self.uav_manager.size()
        assigned_tasks = sum(1 for coalition in coalitions.values() if coalition.uav_ids)
        total_tasks = len(coalitions)

        logger.info(
            f"Allocation summary: {assigned_uavs}/{total_uavs} UAVs assigned, {assigned_tasks}/{total_tasks} tasks covered"
        )


class AuctionBiddingSolverAdvanced(AuctionBiddingSolver):
    """
    Advanced version of the Auction-based Task Allocation Solver

    This version extends the base AuctionBiddingSolver with:
    - Dynamic coalition size limits based on task requirements
    - Simulated annealing for escaping local optima
    - Adaptive bid decay based on allocation progress
    - Task priority handling
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)

        # Additional parameters for advanced features
        self.use_simulated_annealing = True
        self.initial_temperature = 1.0
        self.cooling_rate = 0.95
        self.consider_task_dependencies = True
        self.adaptive_coalition_sizing = True

    @classmethod
    def type_name(cls):
        return "AuctionBiddingAdvanced"

    def run_allocate(self):
        """
        Run the advanced auction-based task allocation algorithm.
        """
        # Initialize coalitions for each task
        coalitions = {
            task.id: Coalition(task_id=task.id)
            for task in self.task_manager.get_all()
            if task.id != self.task_manager.free_uav_task_id
        }

        # Track best bids for each UAV
        uav_best_bids = {uav.id: None for uav in self.uav_manager.get_all()}

        # Track metrics for termination condition
        rounds_without_improvement = 0
        previous_total_utility = 0
        temperature = self.initial_temperature

        # Calculate task priorities based on requirements and time windows
        task_priorities = self._calculate_task_priorities()

        # Main auction loop
        for iteration in range(self.hyper_params.max_iter):
            logger.debug(f"Starting auction iteration {iteration}, temperature={temperature:.4f}")

            # Track if any changes were made in this round
            changes_made = False

            # Process UAVs in random order to avoid bias
            uav_list = list(self.uav_manager.get_all())
            np.random.shuffle(uav_list)

            # Determine max coalition sizes for this round if using adaptive sizing
            max_coalition_sizes = (
                self._determine_coalition_sizes(coalitions)
                if self.adaptive_coalition_sizing
                else None
            )

            # Each UAV submits bids
            for uav in uav_list:
                # Calculate bids for each task, considering priorities
                bids = self._calculate_bids_for_uav_advanced(
                    uav, coalitions, task_priorities, max_coalition_sizes
                )

                if not bids:
                    continue

                # Find the best bid
                best_bid = max(bids, key=lambda b: b.value)

                # Get current task assignment
                current_task_id = self._get_current_task_for_uav(uav.id, coalitions)

                # Determine if UAV should be reassigned, potentially using simulated annealing
                should_reassign = self._should_reassign_uav_advanced(
                    uav.id, current_task_id, best_bid, uav_best_bids, temperature
                )

                if should_reassign:
                    # Remove UAV from current coalition if assigned
                    if current_task_id is not None:
                        self._remove_uav_from_coalition(uav.id, current_task_id, coalitions)

                    # Add UAV to new coalition
                    self._add_uav_to_coalition(best_bid, coalitions)

                    # Update UAV's best bid
                    uav_best_bids[uav.id] = best_bid
                    changes_made = True

            # Calculate total utility of current allocation
            total_utility = self._calculate_total_utility(coalitions)

            # Update termination metrics
            if total_utility > previous_total_utility * (1 + self.min_bid_improvement):
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            previous_total_utility = total_utility

            # Cool down temperature for simulated annealing
            temperature *= self.cooling_rate

            # Log progress
            logger.info(
                f"Iteration {iteration}: Total utility = {total_utility:.2f}, "
                + f"Changes made = {changes_made}, "
                + f"Rounds without improvement = {rounds_without_improvement}"
            )

            # Check termination conditions
            if rounds_without_improvement >= self.max_rounds_without_improvement:
                logger.info(f"Auction converged after {iteration + 1} iterations")
                break

        # Apply final allocation to coalition manager
        self._apply_allocation_to_coalition_manager(coalitions)

        return coalitions

    def _calculate_task_priorities(self) -> Dict[int, float]:
        """Calculate priorities for tasks based on requirements and time constraints."""
        priorities = {}

        for task in self.task_manager.get_all():
            if task.id == self.task_manager.free_uav_task_id:
                continue

            # Priority factors:
            # 1. Time window urgency (earlier deadline = higher priority)
            # 2. Resource requirements (more resources = higher priority)
            # 3. Threat level (higher threat = higher priority)

            # Normalize time window (assuming all time windows start at 0)
            time_urgency = 1.0
            if task.time_window and task.time_window[1] > 0:
                max_time = max(
                    t.time_window[1]
                    for t in self.task_manager.get_all()
                    if t.id != self.task_manager.free_uav_task_id and t.time_window
                )
                time_urgency = 1.0 - (task.time_window[1] / max_time) if max_time > 0 else 1.0

            # Resource requirements
            resource_intensity = np.sum(task.required_resources) / self.hyper_params.resources_num

            # Combine factors
            priority = 0.5 * time_urgency + 0.3 * resource_intensity + 0.2 * task.threat
            priorities[task.id] = priority

        return priorities

    def _determine_coalition_sizes(self, coalitions: Dict[int, Coalition]) -> Dict[int, int]:
        """Dynamically determine maximum coalition sizes based on task requirements."""
        max_sizes = {}

        for task_id, coalition in coalitions.items():
            task = self.task_manager.get(task_id)

            # Base size on resource requirements
            total_required = np.sum(task.required_resources)
            avg_uav_resources = np.mean(
                [np.sum(uav.resources) for uav in self.uav_manager.get_all()]
            )

            # Estimate needed UAVs based on resources
            estimated_size = int(
                np.ceil(total_required / (avg_uav_resources * 0.7))
            )  # 0.7 efficiency factor

            # Constrain to reasonable limits
            max_size = max(1, min(estimated_size, self.max_coalition_size))
            max_sizes[task_id] = max_size

        return max_sizes

    def _calculate_bids_for_uav_advanced(
        self,
        uav: UAV,
        coalitions: Dict[int, Coalition],
        task_priorities: Dict[int, float],
        max_coalition_sizes: Dict[int, int] = None,
    ) -> List[Bid]:
        """Calculate bids with advanced features like task priorities and dynamic sizing."""
        bids = []

        for task in self.task_manager.get_all():
            # Skip the free UAV task
            if task.id == self.task_manager.free_uav_task_id:
                continue

            # Check coalition size limit
            max_size = (
                max_coalition_sizes.get(task.id, self.max_coalition_size)
                if max_coalition_sizes
                else self.max_coalition_size
            )
            if len(coalitions[task.id].uav_ids) >= max_size:
                continue

            # Calculate marginal utility
            utility_before = self._calculate_coalition_utility(task, coalitions[task.id].uav_ids)

            # Create temporary coalition with this UAV added
            temp_coalition = coalitions[task.id].uav_ids.copy()
            temp_coalition.add(uav.id)

            utility_after = self._calculate_coalition_utility(task, temp_coalition)

            # Marginal utility is the improvement this UAV brings
            marginal_utility = utility_after - utility_before

            # Apply task priority as a multiplier
            priority_factor = task_priorities.get(task.id, 1.0)
            weighted_utility = marginal_utility * (1.0 + priority_factor)

            # Only create a bid if the UAV improves the coalition
            if weighted_utility > 0:
                bids.append(Bid(uav_id=uav.id, task_id=task.id, value=weighted_utility))

        return bids

    def _should_reassign_uav_advanced(
        self,
        uav_id: int,
        current_task_id: int,
        new_bid: Bid,
        uav_best_bids: Dict[int, Bid],
        temperature: float,
    ) -> bool:
        """Determine if UAV should be reassigned, with simulated annealing option."""
        # Standard logic first
        standard_decision = super()._should_reassign_uav(
            uav_id, current_task_id, new_bid, uav_best_bids
        )

        # If not using simulated annealing or standard decision is to reassign, return that
        if not self.use_simulated_annealing or standard_decision:
            return standard_decision

        # Otherwise, use simulated annealing to potentially escape local optima
        if current_task_id is not None:
            # Get previous best bid
            previous_bid = uav_best_bids.get(uav_id)

            if previous_bid:
                # Calculate acceptance probability based on temperature and utility difference
                utility_diff = new_bid.value - previous_bid.value

                # If new bid is worse, use probability to decide
                if utility_diff < 0:
                    acceptance_prob = np.exp(utility_diff / temperature)
                    return np.random.random() < acceptance_prob

        return False
