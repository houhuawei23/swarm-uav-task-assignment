import time
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Type, List

from framework.base import HyperParams
from framework.uav import UAV, UAVManager, generate_uav_list
from framework.task import Task, TaskManager, generate_task_list
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
from framework.utils import evaluate_assignment

from . import (
    auction_solver,
    csci2024,
    iros2024,
    icra2024,
    enum_solver,
    milp_solver,
    nlp_solver,
    centralized_solver,
    distributed_solver,
    brute_force_solver,
)


def get_SolverType(choice: str) -> Type[MRTASolver]:
    if choice == "enum":
        return enum_solver.EnumerationSolver
    elif choice == "brute_force":
        return brute_force_solver.BruteForceSearchSolver
    elif choice == "csci":
        return csci2024.ChinaScience2024_CoalitionFormationGame
    elif choice == "iros":
        return iros2024.IROS2024_CoalitionFormationGame
    elif choice == "iros2":
        return iros2024.IROS2024_CoalitionFormationGame_2
    elif choice == "icra":
        return icra2024.ICRA2024_CoalitionFormationGame
    elif choice == "auction":
        return auction_solver.AuctionBiddingSolverAdvanced
    elif choice == "auction_kimi":
        return auction_solver.AuctionBiddingSolverKimi
    elif choice == "milp":
        return milp_solver.MILPSolver
    elif choice == "milp_pyomo":
        return milp_solver.MILPSolverPyomo
    elif choice == "nlp_scipy":
        return nlp_solver.NLPSolverScipy
    elif choice == "nlp_pyomo":
        return nlp_solver.NLPSolverPyomo
    elif choice == "centralized":
        return centralized_solver.CentralizedSolver
    elif choice == "distributed":
        return distributed_solver.DistributedSolver
    else:
        raise ValueError("Invalid choice")

choices_short2long = {
    "csci": "CSCI2024_Xue",
    "iros": "IROS2024_LiwangZhang",
    "icra": "ICRA2024_LiwangZhang",
    "milp": "MILP",
    "nlp_pyomo": "MINLP_Pyomo",
    "centralized": "Centralized",
    "distributed": "Distributed"
}

def get_SolverTypes(choice_list: List[str]) -> List[Type[MRTASolver]]:
    all_choices = ["csci", "iros", "icra", "milp", "nlp_pyomo", "centralized", "distributed"]
    if len(choice_list) == 1 and choice_list[0] == "all":
        return [get_SolverType(choice) for choice in all_choices]
    else:
        return [get_SolverType(choice) for choice in choice_list]


def run_solver(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    choice,
    output_path,
    result_queue: Queue = None,
) -> CoalitionManager:
    print("---")
    coalition_manager: CoalitionManager = CoalitionManager(
        uav_manager.get_ids(), task_manager.get_ids()
    )

    solver = get_SolverType(choice)(uav_manager, task_manager, coalition_manager, hyper_params)

    start_time = time.time()
    solver.run_allocate()
    end_time = time.time()
    elapsed_time = end_time - start_time

    if result_queue is not None:
        result_queue.put(elapsed_time)

    print(f"{choice} Result: {coalition_manager}")

    eval_reuslt = evaluate_assignment(
        uav_manager, task_manager, coalition_manager.task2coalition, hyper_params.resources_num
    )
    print(f"Eval Result: {eval_reuslt}")
    coalition_manager.plot_map(
        uav_manager, task_manager, hyper_params, output_path, plot_unassigned=True
    )

    return coalition_manager


def multi_processes_run(
    uav_manager,
    task_manager,
    hyper_params: HyperParams,
    choice: str,
    output_path: str,
    timeout: float,
):
    q1 = Queue()  # 创建队列用于传递返回值
    q2 = Queue()
    # 启动两个进程
    p1 = Process(
        target=run_solver,
        args=(uav_manager, task_manager, hyper_params, choice, output_path, q1),
    )
    p2 = Process(
        target=run_solver,
        args=(uav_manager, task_manager, hyper_params, choice, output_path, q2),
    )

    p1.start()
    p2.start()
    p1.join(timeout=timeout)  # in seconds
    p2.join(timeout=timeout)
    if p1.is_alive():
        print("Enumeration process did not finish in time. Terminating...")
        p1.terminate()
        p1.join()
    else:
        elapsed_time = q1.get()
        print(f"Enumeration Elapsed Time: {elapsed_time}")

    if p2.is_alive():
        print("Coalition Game process did not finish in time. Terminating...")
        p2.terminate()
        p2.join()
    else:
        elapsed_time = q2.get()
        print(f"Coalition Game Elapsed Time: {elapsed_time}")
