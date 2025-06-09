from typing import List, Tuple, Dict, Type
import random
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Lock
import time
import logging
import signal
import sys
from concurrent.futures import ProcessPoolExecutor

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
import framework.utils as utils

from .utils import MRTA_CFG_Model, get_connected_components_uavid, MRTA_CFG_Model_HyperParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DistributedMultiProcessSolver")

log_level: LogLevel = LogLevel.SILENCE


@dataclass
class Message:
    """Message class for communication between UAV processes."""
    uav_id: int
    changed: bool
    uav_update_step_dict: Dict[int, int]
    task2coalition: Dict[int, List[int]]
    uav2task: Dict[int, int]


class AutoUAVProcess(UAV):
    """Process-safe version of AutoUAV that runs in its own process."""
    
    def __init__(
        self,
        id: int,
        position: List[float] | np.ndarray,
        resources: List[float] | np.ndarray,
        value: float,
        max_speed: float,
        mass: float | None = 1.0,
        fly_energy_per_time: float = random.uniform(1, 3),
        hover_energy_per_time: float = random.uniform(1, 3),
    ):
        super().__init__(
            id,
            position,
            resources,
            value,
            max_speed,
            mass,
            fly_energy_per_time,
            hover_energy_per_time,
        )
        self.changed = False
        self.uav_update_step_dict = {}
        self.message_queue = Queue()
        self.lock = Lock()
        self.stop_event = Event()
        self.process = None

    def init(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        """Initialize the UAV with necessary managers and parameters."""
        with self.lock:
            self.uav_manager = uav_manager
            self.task_manager = task_manager
            self.coalition_manager = coalition_manager
            self.hyper_params = hyper_params
            self.uav_update_step_dict = {uav_id: 0 for uav_id in uav_manager.get_ids()}
            
            map_shape = utils.calculate_map_shape_on_mana(uav_manager, task_manager)
            max_distance = max(map_shape)
            max_uav_value = max(uav.value for uav in uav_manager.get_all())
            self.model_hparams = MRTA_CFG_Model_HyperParams(
                max_distance=max_distance,
                max_uav_value=max_uav_value,
                w_sat=15.0,
                w_waste=1,
                w_dist=25,
                w_threat=1,
            )

    def start(self):
        """Start the UAV's process."""
        self.process = Process(target=self._run, daemon=True)
        self.process.start()

    def stop(self):
        """Stop the UAV's process."""
        self.stop_event.set()
        if self.process:
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()

    def _run(self):
        """Main process loop for the UAV."""
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"UAV {self.id} received signal {signum}")
            self.stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        while not self.stop_event.is_set():
            try:
                # Process messages from queue
                while not self.message_queue.empty():
                    msg = self.message_queue.get_nowait()
                    self.receive_and_update([msg])

                # Try to divert to a better task
                self.try_divert()

                # Sleep briefly to prevent CPU overuse
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in UAV {self.id} process: {str(e)}")
                self.stop_event.set()
                break

    def send_msg(self) -> Message:
        """Create and return a message with current state."""
        with self.lock:
            return Message(
                uav_id=self.id,
                changed=self.changed,
                uav_update_step_dict=deepcopy(self.uav_update_step_dict),
                task2coalition=deepcopy(self.coalition_manager.get_task2coalition()),
                uav2task=deepcopy(self.coalition_manager.get_uav2task()),
            )

    def receive_and_update(self, msgs: List[Message]) -> bool:
        """Process received messages and update local state."""
        receive_changed = False
        with self.lock:
            for msg in msgs:
                for msg_uav_id, msg_uav_update_step in msg.uav_update_step_dict.items():
                    if msg_uav_id not in self.uav_update_step_dict:
                        self.uav_update_step_dict[msg_uav_id] = 0
                    if self.uav_update_step_dict[msg_uav_id] < msg_uav_update_step:
                        self.coalition_manager.unassign(msg_uav_id)
                        self.coalition_manager.assign(msg_uav_id, msg.uav2task[msg_uav_id])
                        self.uav_update_step_dict[msg_uav_id] = msg_uav_update_step
                        receive_changed = True

            self.changed = self.changed or receive_changed
        return receive_changed

    def try_divert(self, prefer: str = "cooperative") -> bool:
        """Try to divert to a better task."""
        with self.lock:
            task_ids = self.task_manager.get_ids().copy()
            divert_changed = False
            
            for taskj_id in task_ids:
                taski_id = self.coalition_manager.get_taskid(self.id)
                if taski_id == taskj_id:
                    continue

                taski = self.task_manager.get(taski_id)
                taskj = self.task_manager.get(taskj_id)

                prefer_func = MRTA_CFG_Model.get_prefer_func(prefer)
                if prefer_func(
                    uav=self,
                    task_p=taski,
                    task_q=taskj,
                    uav_manager=self.uav_manager,
                    task_manager=self.task_manager,
                    coalition_manager=self.coalition_manager,
                    resources_num=self.hyper_params.resources_num,
                    model_hparams=self.model_hparams,
                ):
                    self.coalition_manager.unassign(self.id)
                    self.coalition_manager.assign(self.id, taskj_id)
                    divert_changed = True
                    self.uav_update_step_dict[self.id] += 1

            self.changed = self.changed or divert_changed
            return divert_changed

    def broadcast_message(self, message: Message):
        """Add a message to the UAV's message queue."""
        self.message_queue.put(message)


class DistributedSolver_MultiProcess(MRTASolver):
    """Multi-process version of the distributed solver."""

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        self.uav_processes: Dict[int, AutoUAVProcess] = {}
        self.message_queues: Dict[int, Queue] = {}
        self.stop_event = Event()

    @staticmethod
    def uav_type() -> Type:
        return AutoUAVProcess

    @classmethod
    def type_name(cls):
        return "DistributedMultiProcess"

    def init_allocate(self, components: List[List[int]], debug=False):
        """Initialize UAV processes and their allocations."""
        task_ids = self.task_manager.get_ids()
        init_assignment = {task_id: [] for task_id in task_ids}

        # Initialize random assignments
        for component in components:
            for uav_id in component:
                task_id = random.choice(task_ids)
                init_assignment[task_id].append(uav_id)

        # Initialize UAV processes
        for component in components:
            component_uav_manager = UAVManager(
                [self.uav_manager.get(uav_id) for uav_id in component]
            )
            
            for uav_id in component:
                uav = AutoUAVProcess(
                    id=uav_id,
                    position=self.uav_manager.get(uav_id).position,
                    resources=self.uav_manager.get(uav_id).resources,
                    value=self.uav_manager.get(uav_id).value,
                    max_speed=self.uav_manager.get(uav_id).max_speed,
                )
                
                uav_coalition_manager = CoalitionManager(
                    self.uav_manager.get_ids(), task_ids
                )
                
                uav.init(
                    component_uav_manager,
                    self.task_manager,
                    uav_coalition_manager,
                    self.hyper_params,
                )
                
                uav_coalition_manager.update_from_assignment(
                    init_assignment, uav.uav_manager
                )
                
                self.uav_processes[uav_id] = uav
                self.message_queues[uav_id] = Queue()

    def run_allocate(self):
        """Run the multi-process allocation algorithm."""
        comm_distance = self.hyper_params.map_shape[0]
        uav_list = self.uav_manager.get_all()
        components = get_connected_components_uavid(uav_list, comm_distance)

        # Initialize UAV processes
        self.init_allocate(components)

        # Start all UAV processes
        for uav in self.uav_processes.values():
            uav.start()

        try:
            max_not_changed_iter = 5
            not_changed_iter_cnt = 0
            iter = 0

            while True:
                iter += 1
                if not_changed_iter_cnt > max_not_changed_iter:
                    logger.info(f"Reached max_not_changed_iter {max_not_changed_iter}")
                    break
                if iter > self.hyper_params.max_iter:
                    logger.info(f"Reached max iter {self.hyper_params.max_iter}")
                    break

                total_changed = False

                # Process each component
                for component in components:
                    # Sample UAVs from component
                    sample_rate = 1/3
                    rec_sample_size = max(1, int(len(component) * sample_rate))
                    sampled_uavids = random.sample(component, rec_sample_size)

                    # Collect messages from sampled UAVs
                    messages = []
                    for uav_id in sampled_uavids:
                        uav = self.uav_processes[uav_id]
                        uav.changed = False
                        uav.try_divert()
                        msg = uav.send_msg()
                        messages.append(msg)

                    # Broadcast messages to all UAVs in component
                    for uav_id in component:
                        uav = self.uav_processes[uav_id]
                        for msg in messages:
                            uav.broadcast_message(msg)

                    component_changed = any(msg.changed for msg in messages)
                    total_changed = total_changed or component_changed

                # Leader communication
                leader_messages = []
                for component in components:
                    if not component:
                        continue
                    leader_uav_id = component[0]
                    leader_uav = self.uav_processes[leader_uav_id]
                    msg = leader_uav.send_msg()
                    leader_messages.append(msg)

                # Broadcast leader messages
                for component in components:
                    if not component:
                        continue
                    leader_uav_id = component[0]
                    leader_uav = self.uav_processes[leader_uav_id]
                    for msg in leader_messages:
                        leader_uav.broadcast_message(msg)

                leaders_changed = any(msg.changed for msg in leader_messages)
                total_changed = total_changed or leaders_changed

                if not total_changed:
                    not_changed_iter_cnt += 1
                else:
                    not_changed_iter_cnt = 0

                if log_level >= LogLevel.INFO:
                    logger.info(f"Iteration {iter} completed")
                    for component in components:
                        for uav_id in component:
                            uav = self.uav_processes[uav_id]
                            logger.info(uav.debug_info())

        finally:
            # Stop all UAV processes
            for uav in self.uav_processes.values():
                uav.stop()

        # Update final coalition state
        for component in components:
            if not component:
                continue
            leader_uav_id = component[0]
            leader_uav = self.uav_processes[leader_uav_id]
            self.coalition_manager.merge_coalition_manager(leader_uav.coalition_manager)

        if log_level >= LogLevel.INFO:
            logger.info("Final coalition:")
            self.coalition_manager.format_print() 