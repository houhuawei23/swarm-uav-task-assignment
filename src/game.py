from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from uav import UAV, UAVManager
from task import Task, TaskManager, get_resources_weights
from coalition import CoalitionManager
from utils import (
    calculate_path_cost,
    calculate_resource_contribution,
    calculate_threat_cost,
    calculate_obtained_resources,
)
from base import HyperParams
from dataclasses import dataclass, field

np.set_printoptions(precision=2)


def cal_resource_contribution(
    uav: UAV, task: Task, task_collation: List[int], task_obtained_resources: List[float], debug=False
) -> float:
    """Calculates the resource contribution of a UAV to a task.

    task_obtained_resources = [...]
    now_required_resources = [1, 2, -3]
    np.maximum(now_required_resources, 0) = [1, 2, 0]
    uav.resources = [2, 3, 4]
    uav.resources - np.maximum(now_required_resources, 0) = [0, 1, 4]
    surplus_resources_sum_of_uav = 0 + 1 + 4 = 5
    """
    # 资源贡献计算
    # I = uav.resources + satisfied, paper meaning this???

    # +is required, -is surplus
    task_obtained_resources_copy = task_obtained_resources.copy()

    # 如果 uav 在 task 的联盟中，则减去 uav 的贡献
    if uav.id in task_collation:
        task_obtained_resources_copy -= uav.resources

    now_required_resources = task.required_resources - task_obtained_resources_copy
    surplus_resources_sum_of_uav = sum(np.maximum(uav.resources - np.maximum(now_required_resources, 0), 0))
    if debug:
        print(f"abtained_resources: {task_obtained_resources_copy}")

    Kj = get_resources_weights(task.required_resources, task_obtained_resources_copy)  # 表征 task 对资源的偏好
    I = uav.resources  # only consider uav's own resources
    O = surplus_resources_sum_of_uav
    P = 0.5

    if debug:
        print(f"Kj: {Kj}, I: {I}, O: {O: .2f}, P: {P: .2f}")
    val = calculate_resource_contribution(Kj, I, O, P)
    return val


def cal_uav_task_benefit(
    uav: UAV,
    task: Task,
    task_collation: List[int],
    task_obtained_resources: List[float],
    hyper_params: HyperParams,
    debug=False,
) -> float:
    """Calculates the total benefit of assigning a UAV to a task.

    r(ui, tj) = alpha * val(ui, tj) + beta * cost(ui, tj) - gamma * risk(ui, tj)

    The total benefit is a weighted sum of resource contribution, path cost, and threat cost.
    The weights (alpha, beta, gamma) are used to balance the importance of each factor.

    # epsilon(uav, task) =

    Args:
        uav: The UAV object representing the unmanned aerial vehicle.
        task: The task object representing the task to be assigned.

    Returns:
        float: The total benefit of assigning the UAV to the task.
    """
    if debug:
        print(f"cal_uav_task_benefit(uav=u{uav.id}, task=t{task.id})")
    # 计算资源贡献
    resource_contribution = cal_resource_contribution(uav, task, task_collation, task_obtained_resources, debug)
    # 计算路径成本
    path_cost = calculate_path_cost(uav, task, resource_contribution, hyper_params.map_shape, hyper_params.mu)
    # 计算威胁代价
    threat_cost = calculate_threat_cost(uav, task)
    # 总收益
    total_benefit = (
        hyper_params.alpha * resource_contribution + hyper_params.beta * path_cost - hyper_params.gamma * threat_cost
    )
    # check time window
    min_uav_fly_time = uav.position.distance_to(task.position) / uav.max_speed
    if min_uav_fly_time > task.time_window[1]:
        total_benefit = -100
    # debug = True
    if debug:
        print(
            f"  val={resource_contribution:.2f}, path_cost={path_cost:.2f}, threat_cost={threat_cost:.2f}, benefit={total_benefit:.2f}"
        )
        print("cal_uav_task_benefit finished")
    return total_benefit


def cal_task_on_given_coalition_benefit(
    task: Task, given_coalition: List[int], uav_manager: UAVManager, hyper_params: HyperParams, debug=False
) -> float:
    """
    the benefit of a coalition for task = sum of the benefit of each UAV in the coalition.
    R(ctj) = sum(cal_benefit(ui, tj)) for ui in ctj
    """
    if debug:
        print(f"cal_task_coalition_utility(task=t{task.id}, coalition={given_coalition})")
    utility = 0.0
    for uav_id in given_coalition:  # ??? coalition=[u1, u2] on t2, 效用不应该简单叠加吧！！
        uav = uav_manager.get(uav_id)
        obtained_resources = calculate_obtained_resources(given_coalition, uav_manager, hyper_params.resources_num)

        benefit = cal_uav_task_benefit(uav, task, given_coalition, obtained_resources, hyper_params)
        utility += benefit
    if debug:
        print(f"  utility={utility:.2f}")
        print("cal_task_coalition_utility finished")
    return utility


@dataclass
class CoalitionFormationGame:
    """Represents a coalition formation game where UAVs are assigned to tasks based on benefits.

    The game involves forming coalitions of UAVs for each task, considering factors such as resource contribution,
    path cost, and threat. The goal is to maximize the overall benefit of the assignments.

    G = (U, E, epsilon, R)

    where:
        U: is a set of UAVs.
        E: is the task set selected by the UAV, equivalent to the task set T.
        epsiolon: is a utility function that evaluates the benefits of a drone.
        R: is a function that evaluates the utility of individual coalitions.

    R(ctj) 任务 tj 的联盟效用: R(ctj) = sum_{ui in ctj} [r(ui, tj)], 即执行该任务的所有无人机的效用之和。
    SR 任务分配问题的总收益: SR = sum_{ctj in CS} R(ctj)
    ui 执行 tj 的收益: r(ui, tj) = alpha * val(ui, tj) + beta * dist(ui, tj) - gamma * risk(ui, tj)
    epsilon(eu, E_{-ui}) 无人机效用函数: epsilon(eu, E_{-ui}) = R(ctj) - R(ctj - {ui})

    Game Goal: 最大化总收益 SR
    约束条件:
        (1) 速度约束;
        (2) 任务时效性约束;
        (3) 无人机执行任务模式约束: 同一时间只允许一个无人机执行一个任务。


    Attributes:
        uavs (list): A list of UAV objects available for task assignment.
        tasks (list): A list of Task objects that need to be completed.
        alpha (float): Weight for resource contribution in the benefit calculation.
        beta (float): Weight for path cost in the benefit calculation.
        gamma (float): Weight for threat in the benefit calculation.
        coalitions (dict): A dictionary mapping task IDs to lists of UAVs assigned to them. The key `None` represents
                           UAVs that are not assigned to any task.
    """

    uav_manager: UAVManager
    task_manager: TaskManager
    coalition_manager: CoalitionManager
    hyper_params: HyperParams

    def cal_benefit_matrix(self, unassigned_uav_ids: List[int], task_ids: List[int], debug=False) -> np.ndarray:
        """Calculates the benefit matrix for UAVs and tasks.
        计算每个无人机对每个任务的收益矩阵

        The benefit matrix is a 2D array where each element represents the benefit of assigning a specific UAV to a specific task.
        The benefit is calculated based on resource contribution, path cost, and threat cost.

        Returns:
            np.ndarray: A 2D numpy array of shape (n, m), where n is the number of UAVs and m is the number of tasks.
                        Each element benefit_matrix[i, j] represents the benefit of assigning UAV i to task j.
        """
        n = len(unassigned_uav_ids)
        m = len(task_ids)
        benefit_matrix = np.zeros((n, m))

        for i, uav_id in enumerate(unassigned_uav_ids):
            for j, task_id in enumerate(task_ids):
                uav = self.uav_manager.get(uav_id)
                task = self.task_manager.get(task_id)
                coalition = self.coalition_manager.get_coalition(task_id)
                obtained_resources = calculate_obtained_resources(
                    coalition, self.uav_manager, self.hyper_params.resources_num
                )

                benefit_matrix[i, j] = cal_uav_task_benefit(uav, task, coalition, obtained_resources, self.hyper_params)
        # debug = True
        if debug:
            print("Benefit Matrix:")
            print(benefit_matrix)
        return benefit_matrix

    def cal_uav_utility_on_colition(self, uav: UAV, task: Task, coalition: List[int], debug=False) -> float:
        coalition_copy = coalition.copy()
        if uav.id in coalition:  # uav already in coalition
            coalition_copy.remove(uav.id)

        u_not_have = cal_task_on_given_coalition_benefit(task, coalition_copy, self.uav_manager, self.hyper_params)
        coalition_copy.append(uav.id)
        u_have = cal_task_on_given_coalition_benefit(task, coalition_copy, self.uav_manager, self.hyper_params)
        utility = u_have - u_not_have
        return utility

    def cal_sum_utility(self, debug=False) -> float:
        """Calculates the total utility of the current coalition assignments.
        SR = sum_{ctj in CS} R(ctj)
        """
        total_utility = 0.0
        for task in self.task_manager:
            total_utility += cal_task_on_given_coalition_benefit(
                task, self.coalition_manager.get_coalition(task.id), self.uav_manager, self.hyper_params
            )
        if debug:
            print(f"Total Utility: {total_utility:.2f}")
        return total_utility

    def match_tasks(self, benefit_matrix, unassigned_uav_ids, task_ids, debug=False):
        """Matches UAVs to tasks based on the benefit matrix.

        This function uses the Hungarian algorithm (linear sum assignment) to find the optimal assignment
        of UAVs to tasks that maximizes the total benefit. UAVs are assigned to tasks only if the benefit
        is positive. The coalitions dictionary is updated to reflect the assignments.

        Algorithm Steps:
        1. Compute the benefit matrix using `calculate_benefit_matrix`.
        2. Use the Hungarian algorithm to find the optimal assignment.
        3. Assign UAVs to tasks if the benefit is positive.
        4. Update the coalitions dictionary to reflect the assignments.

        每个任务匹配到一个无人机
        """
        # 最大加权匹配
        row_ind, col_ind = linear_sum_assignment(benefit_matrix, maximize=True)
        # print(f"row_ind: {row_ind}, col_ind: {col_ind}")
        have_assigned = False
        for uav_idx, task_idx in zip(row_ind, col_ind):
            print(unassigned_uav_ids, task_ids)
            # print(f"uav_idx: {uav_idx}, task_idx: {task_idx}")
            # print("here", unassigned_uav_ids[uav_idx])
            uav: UAV = self.uav_manager.get(unassigned_uav_ids[uav_idx])
            task: Task = self.task_manager.get(task_ids[task_idx])
            if debug:
                print(f"uav_idx: {uav_idx}, task_idx: {task_idx}")

            if benefit_matrix[uav_idx, task_idx] > 0:
                if debug:
                    print(f"Assigning u{uav.id} to t{task.id}")
                # 6. update the required resources of task
                # task_obtained_resources += uav.resources
                self.coalition_manager.assign(uav, task)
                have_assigned = True
        return have_assigned

    def check_stability(self, debug=False):
        """Checks the stability of the current coalition assignments.

        A coalition is considered stable if no UAV can improve its benefit by switching to another task.
        This function iterates through all UAVs and tasks to check if any UAV can achieve a higher benefit
        by being reassigned to a different task. If such a case is found, the UAV is reassigned, and the
        coalition is marked as unstable.

        Returns:
            bool: True if the coalition is stable, False otherwise.
        """
        stable = True
        # 遍历所有任务的所有已分配的无人机
        for taski in self.task_manager:
            # print(
            #     f"Checking stability for task {task.id}, coalition: {self.coalition_manager[task.id]}"
            # )
            for uav_id in self.coalition_manager.get_coalition(taski.id):
                uav = self.uav_manager.get(uav_id)
                taski_coalition = self.coalition_manager.get_coalition(taski.id)
                cur_utility = self.cal_uav_utility_on_colition(uav, taski, taski_coalition.copy(), debug=debug)

                if debug:
                    print(f"Cur utility: u{uav.id}-t{taski.id}={cur_utility}")

                for taskj in self.task_manager:
                    if taskj.id != taski.id:
                        taskj_coalition = self.coalition_manager.get_coalition(taskj.id)

                        move_to_taskj_utility = self.cal_uav_utility_on_colition(
                            uav, taskj, taskj_coalition.copy(), debug=debug
                        )

                        if debug:
                            print(f"utility: u{uav.id}-t{taskj.id}={move_to_taskj_utility}")
                        if move_to_taskj_utility > cur_utility:
                            self.coalition_manager.unassign(uav)  # 无人机退出原联盟
                            stable = False
                            break

        return stable

    def run(self, debug=False):
        """Runs the coalition formation game until a stable assignment is achieved.

        This function repeatedly matches UAVs to tasks and checks for stability until no further
        improvements can be made. The process terminates when the coalition assignments are stable.

        Returns:
            dict: The final coalitions dictionary mapping task IDs to lists of assigned UAVs.

        1. Initial task allocation result.
        2. Calculate or update the benefit matric.
        3. Matching based on maximum weighed principle.
        4. Update the required resources of task, design rules for profit checking.
        5. Obtain the final stable result of this layer.
        6. Update the carried resources and requirements of tasks.
        7. Requirements of a certain task are an empty set.
            1. if true, jump to 2.
            2. else go to 8.
        8. Exit.When all tasks exit, obtain the final coalition structure.
        """
        iter_cnt = 0
        while True:
            # in once iter, try to assign one uav to each task
            print(f"Iteration {iter_cnt} begin.")
            print(f"Cur coalition set: {self.coalition_manager}")
            if iter_cnt >= self.hyper_params.max_iter:
                print("Max iterations reached, may have dead loop")
                break
            iter_cnt += 1
            # 2. calculate the benefit matrix
            unassigned_uav_ids = self.coalition_manager.get_unassigned_uav_ids().copy()
            task_ids = self.task_manager.get_ids()

            benefit_matrix = self.cal_benefit_matrix(unassigned_uav_ids, task_ids, debug=debug)
            print(unassigned_uav_ids)
            print(task_ids)
            print(benefit_matrix)
            # 3. matching based on maximum weighed principle (based on benefit matrix)
            if not self.match_tasks(benefit_matrix, unassigned_uav_ids, task_ids, debug=debug):
                print("No more UAVs can be assigned to tasks. Over, break.")
                break

            if self.check_stability(debug=False):
                print(f"check_stability True, Iteration {iter_cnt} Assign Valid.")
                # print(f"Cur coalition set: {self.coalition_manager}")
            else:
                print(f"check_stability False, Iteration {iter_cnt} Assign Invalid.")
                # print(f"Cur coalition set: {self.coalition_manager}")

        return self.coalition_manager


def test_calculate_resource_contribution():
    # 示例权重矩阵和资源数据
    weight_matrix = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    constant_P = 0.1

    # 示例数据
    task_index = 0
    resource_vector = np.array([10, 20, 30])  # 可用资源数目 i_1, i_2, ..., i_l
    unused_resources = 5  # 未利用资源总数目
    val = calculate_resource_contribution(weight_matrix[task_index], resource_vector, unused_resources, constant_P)
    print(f"Resource contribution: {val}")


def test_game():
    resources_num = 2
    map_shape = [10, 10, 0]
    gamma = 0.1

    uav1 = UAV(id=1, resources=[5, 3], position=[0, 0, 0], value=1, max_speed=10)
    uav2 = UAV(id=2, resources=[3, 4], position=[10, 10, 0], value=1, max_speed=10)
    uav3 = UAV(id=3, resources=[2, 5], position=[20, 20, 0], value=1, max_speed=10)
    uavs = [uav1, uav2, uav3]
    uav_manager = UAVManager(uavs)
    task1 = Task(1, [4, 2], [5, 5, 0], [0, 100], 0.5)
    task2 = Task(2, [3, 3], [15, 15, 0], [0, 100], 0.5)
    tasks = [task1, task2]
    # tasks = [task1]
    task_manager = TaskManager(tasks)

    coalition_manager = CoalitionManager(uav_manager, task_manager)
    game = CoalitionFormationGame(
        uav_manager,
        task_manager,
        coalition_manager,
        resources_num=resources_num,
        map_shape=map_shape,
        gamma=gamma,
    )

    val = cal_resource_contribution(coalition_manager, uav1, task1)
    # cost = game.cal_path_cost(uav1, task1, val)
    # threat = game.cal_threat_cost(uav1, task1)
    # benefit = game.cal_uav_task_benefit(uav1, task1)
    print(f"{uav1}")
    print(f"{task1}")
    print(f"Resource contribution: {val: .2f}")
    # print(f"Path cost: {cost: .2f}")
    # print(f"Threat cost: {threat: .2f}")
    # print(f"Benefit: {benefit: .2f}")

    coalition_manager.plot_map()
    game.run(debug=True)
    coalition_manager.plot_map()


if __name__ == "__main__":
    # test_calculate_resource_contribution()
    test_game()
