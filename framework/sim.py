import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Any
import time

from .base import HyperParams
from .uav import UAV, UAVManager
from .task import Task, TaskManager
from .coalition_manager import CoalitionManager

from dataclasses import dataclass


@dataclass
class SimState:
    time_step: int
    uav_dict_list: list
    elapsed_time: float = 0.0  # Actual elapsed simulation time in seconds


class SimulationEnv:
    """
    Manages the simulation environment, entities, and execution flow.
    模拟uav从初始位置出发，运动到目标Task位置
    模拟 n 步，每一步遍历所有的 uav，运动一步，记录更新后的uav状态
    记录什么：记录"格局"，即每一步的全局状态信息,or SimState
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_mana: CoalitionManager,
        hyper_params: HyperParams,
    ):
        self.uav_manager = uav_manager
        self.task_manager = task_manager
        self.coalition_manager = coalition_mana
        self.hyper_params = hyper_params

        self.time_step = 0
        self.sim_history: List[SimState] = []

        self.simulation_dt = 0.1  # Time step for simulation in seconds
        self.total_elapsed_time = 0.0  # Total elapsed simulation time

        # Track when tasks are completed
        self.task_completion_times: Dict[int, float] = {}
        # Track UAV states (moving, arrived, etc.), uav_id -> {status, target_task, arrival_time, path}
        self.uav_states: Dict[int, Dict[str, Any]] = {}
        
        # Initialize UAV states
        for uav in self.uav_manager.get_all():
            self.uav_states[uav.id] = {
                'status': 'idle',  # idle, moving, arrived
                'target_task': None,
                'arrival_time': None,
                'path': [uav.position.tolist()]  # Track full path history
            }

    def run(self, steps: int = 10, debug_level=0):
        print("--- Simulation Started ---")
        uav_dict_list = self.uav_manager.to_dict_list()
        sim_state = SimState(self.time_step, uav_dict_list, self.total_elapsed_time)
        self.sim_history.append(sim_state)
        
        for _ in range(steps):
            self.simulate_step(debug=(debug_level > 1))
            
            # Print progress information based on debug level
            if debug_level > 0 and self.time_step % max(1, steps//10) == 0:
                completion_rate = len(self.task_completion_times) / max(1, len(self.task_manager.get_all()) - 1)
                print(f"Step {self.time_step}/{steps} - Time: {self.total_elapsed_time:.2f}s - Tasks completed: {len(self.task_completion_times)} ({completion_rate:.0%})")
        
        print("--- Simulation Finished ---")
        print(f"Total simulation time: {self.total_elapsed_time:.2f} seconds")
        print(f"Tasks completed: {len(self.task_completion_times)}/{len(self.task_manager.get_all()) - 1}")
        
        self.visualize_simulation()

    def simulate_step(self, debug=False):
        self.time_step += 1
        self.total_elapsed_time += self.simulation_dt
        
        unassigned_uav_ids = self.coalition_manager.get_unassigned_uav_ids()
        task_positions = {}  # Cache task positions
        
        # Update UAV positions and states
        for uav in self.uav_manager.get_all():
            if uav.id in unassigned_uav_ids:
                # Unassigned UAVs remain idle
                if self.uav_states[uav.id]['status'] != 'idle':
                    self.uav_states[uav.id]['status'] = 'idle'
                    self.uav_states[uav.id]['target_task'] = None
                continue
                
            # Get assigned task
            task_id = self.coalition_manager.get_taskid(uav.id)
            task = self.task_manager.get(task_id)
            
            # Update UAV state if task assignment changed
            if self.uav_states[uav.id]['target_task'] != task_id:
                self.uav_states[uav.id]['target_task'] = task_id
                self.uav_states[uav.id]['status'] = 'moving'
            
            # Cache task position
            if task_id not in task_positions:
                task_positions[task_id] = task.position
            
            # Check if UAV has arrived at task location
            distance = uav.position.distance_to(task.position)
            arrival_threshold = 0.5  # Distance threshold to consider UAV arrived
            
            if distance <= arrival_threshold:
                if self.uav_states[uav.id]['status'] != 'arrived':
                    self.uav_states[uav.id]['status'] = 'arrived'
                    self.uav_states[uav.id]['arrival_time'] = self.total_elapsed_time
                    
                    # Check if all UAVs in coalition have arrived
                    coalition = self.coalition_manager.get_coalition(task_id)
                    all_arrived = all(
                        self.uav_states[coalition_uav_id]['status'] == 'arrived' 
                        for coalition_uav_id in coalition
                    )
                    
                    if all_arrived and task_id not in self.task_completion_times:
                        self.task_completion_times[task_id] = self.total_elapsed_time
                        if debug:
                            print(f"Task {task_id} completed at time {self.total_elapsed_time:.2f}s")
            else:
                # Move UAV towards task with precise physics
                moved_distance = uav.move_to(task.position, self.simulation_dt)
                
                # Record path for visualization
                self.uav_states[uav.id]['path'].append(uav.position.tolist())
                
                if debug:
                    print(f"UAV {uav.id} moved {moved_distance:.2f} units toward Task {task_id}. Distance remaining: {distance:.2f}")

        # Record state
        uav_dict_list = self.uav_manager.to_dict_list()
        sim_state = SimState(self.time_step, uav_dict_list, self.total_elapsed_time)
        self.sim_history.append(sim_state)

    def visualize_simulation(self):
        # Set up a professional-looking figure with improved styling
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 10), dpi=120)
        
        # Create main simulation plot and timeline subplot
        gs = plt.GridSpec(3, 3, figure=fig)
        ax_main = fig.add_subplot(gs[:, :2])  # Main simulation area
        ax_timeline = fig.add_subplot(gs[0, 2])  # Timeline
        ax_stats = fig.add_subplot(gs[1, 2])  # Statistics
        ax_legend = fig.add_subplot(gs[2, 2])  # Legend
        
        # Set background colors
        fig.patch.set_facecolor('white')
        ax_main.set_facecolor('#f8f9fa')
        ax_timeline.set_facecolor('#f8f9fa')
        ax_stats.set_facecolor('#f8f9fa')
        ax_legend.set_facecolor('#f8f9fa')
        
        # Set up main plot
        shapex, shapey, _ = self.hyper_params.map_shape
        ax_main.set_xlim(0, shapex * 1.1)
        ax_main.set_ylim(0, shapey * 1.1)
        ax_main.set_xlabel("X Position", fontsize=12, fontweight='bold')
        ax_main.set_ylabel("Y Position", fontsize=12, fontweight='bold')
        ax_main.set_title("UAV Task Assignment Simulation", fontsize=16, fontweight='bold', pad=15)
        ax_main.grid(True, linestyle='--', alpha=0.3)
        
        # Extract UAV initial positions
        uav_positions = [np.array(uav["position"])[:2] for uav in self.sim_history[0].uav_dict_list]
        uav_positions = np.array(uav_positions) if uav_positions else np.empty((0, 2))
        
        # Extract UAV trajectories over time
        uav_trajectories = []
        for i in range(len(self.sim_history[0].uav_dict_list)):
            trajectory = [np.array(state.uav_dict_list[i]["position"])[:2] for state in self.sim_history]
            uav_trajectories.append(trajectory)
        
        # Extract Task positions and details
        tasks = [task for task in self.task_manager.get_all() if task.id != self.task_manager.free_uav_task_id]
        task_positions = np.array([task.position.xyz[:2] for task in tasks]) if tasks else np.empty((0, 2))
        
        # Create color maps for UAVs and tasks
        uav_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(uav_positions)))
        task_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(task_positions)))
        
        # Plot task positions with enhanced styling
        for i, task in enumerate(tasks):
            x, y = task.position.xyz[:2]
            
            # Add task highlight/glow
            task_glow = plt.Circle((x, y), 1.2, color=task_colors[i], alpha=0.15, zorder=1)
            ax_main.add_patch(task_glow)
            
            # Plot task marker
            ax_main.scatter(x, y, color=task_colors[i], marker='s', s=120, 
                          edgecolor='white', linewidth=1.5, zorder=10)
            
            # Add task label
            ax_main.text(x, y + 0.4, f"Task {task.id}", 
                       ha="center", color='black', fontsize=10,
                       fontweight='medium', bbox=dict(facecolor='white', alpha=0.8, 
                                                   edgecolor=task_colors[i], boxstyle='round,pad=0.2'),
                       zorder=11)
        
        # Initialize UAV scatter plot
        uav_scatter = ax_main.scatter(
            [], [], s=100, 
            edgecolor='white', linewidth=1.5,
            zorder=20
        )
        
        # Initialize trail lines for each UAV
        trail_lines = []
        for i in range(len(uav_positions)):
            line, = ax_main.plot([], [], '-', linewidth=1.5, alpha=0.5, color=uav_colors[i])
            trail_lines.append(line)
        
        # Initialize time indicator
        time_text = ax_main.text(
            0.02, 0.98, '', transform=ax_main.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'),
            zorder=30, ha='left', va='top'
        )
        
        # Set up timeline plot
        ax_timeline.set_title("Simulation Timeline", fontsize=12, fontweight='bold')
        ax_timeline.set_xlabel("Time (seconds)", fontsize=10)
        ax_timeline.set_ylabel("UAV ID", fontsize=10)
        ax_timeline.set_yticks(range(1, len(uav_positions) + 1))
        
        # Prepare timeline data
        timeline_data = np.zeros((len(uav_positions), len(self.sim_history)))
        for uav_idx in range(len(uav_positions)):
            uav_id = self.sim_history[0].uav_dict_list[uav_idx]["id"]
            for t in range(len(self.sim_history)):
                if t > 0:  # Skip first frame
                    # Color based on status: moving or arrived
                    state = self.uav_states[uav_id]
                    if state['status'] == 'arrived' and t * self.simulation_dt >= state['arrival_time']:
                        timeline_data[uav_idx, t] = 2  # Arrived
                    elif state['status'] != 'idle':
                        timeline_data[uav_idx, t] = 1  # Moving
        
        # Plot timeline heatmap
        timeline_times = [state.elapsed_time for state in self.sim_history]
        timeline_img = ax_timeline.imshow(
            timeline_data, aspect='auto', cmap='viridis',
            extent=[0, timeline_times[-1], 0.5, len(uav_positions) + 0.5],
            interpolation='nearest'
        )
        
        # Set up stats plot
        ax_stats.set_title("Completion Statistics", fontsize=12, fontweight='bold')
        ax_stats.axis('off')
        
        # Set up legend
        ax_legend.set_title("Legend", fontsize=12, fontweight='bold')
        ax_legend.axis('off')
        
        # Add legend items
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=uav_colors[0], 
                      markersize=10, label='UAV'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=task_colors[0], 
                      markersize=10, label='Task'),
            plt.Line2D([0], [0], linestyle='-', color=uav_colors[0], 
                      label='UAV Path'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#ccffcc', alpha=0.5, 
                         label='Arrived'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#ffcc99', alpha=0.5, 
                         label='Moving')
        ]
        ax_legend.legend(handles=legend_elements, loc='center', fontsize=10)
        
        # Add stats text
        completed_tasks = len(self.task_completion_times)
        total_tasks = len(tasks)
        stats_text = (
            f"Simulation Summary:\n"
            f"- Time steps: {self.time_step}\n"
            f"- Elapsed time: {self.total_elapsed_time:.2f}s\n"
            f"- Tasks completed: {completed_tasks}/{total_tasks} ({completed_tasks/max(1,total_tasks):.0%})\n"
            f"- UAVs: {len(uav_positions)}\n"
            f"- Average speed: {np.mean([uav.max_speed for uav in self.uav_manager.get_all()]):.2f}"
        )
        
        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        # Animation update function
        def update(frame):
            # Update UAV positions
            current_positions = np.array([uav_trajectories[i][frame] for i in range(len(uav_trajectories))])
            uav_scatter.set_offsets(current_positions)
            
            # Set colors based on UAV task assignment
            colors = []
            for i, uav in enumerate(self.sim_history[frame].uav_dict_list):
                uav_id = uav["id"]
                task_id = self.coalition_manager.get_taskid(uav_id)
                
                if task_id == self.task_manager.free_uav_task_id:
                    # Unassigned UAVs are gray
                    colors.append('gray')
                else:
                    # Assigned UAVs get color based on their task
                    task_idx = [t.id for t in tasks].index(task_id)
                    colors.append(task_colors[task_idx])
            
            uav_scatter.set_color(colors)
            
            # Update UAV trails
            for i in range(len(uav_trajectories)):
                # Show trail of recent positions (last 10 steps)
                start_idx = max(0, frame - 10)
                trail_lines[i].set_data(
                    [uav_trajectories[i][j][0] for j in range(start_idx, frame + 1)],
                    [uav_trajectories[i][j][1] for j in range(start_idx, frame + 1)]
                )
            
            # Update time indicator
            time_text.set_text(f"Time: {self.sim_history[frame].elapsed_time:.2f}s")
            
            return [uav_scatter, time_text] + trail_lines

        # Create animation with progress bar
        from matplotlib.animation import FuncAnimation
        from tqdm.notebook import tqdm
        
        frames = len(self.sim_history)
        interval = 100  # milliseconds between frames
        
        # Create animation
        ani = FuncAnimation(
            fig, update, frames=frames,
            interval=interval, blit=True
        )
        
        # Add timestamp and info
        plt.figtext(
            0.01, 0.01, 
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=8, 
            color='gray'
        )
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Show animation
        plt.show()
        
        # Optional: Save animation as GIF or MP4
        # ani.save('simulation.mp4', writer='ffmpeg', fps=10, dpi=100)
        
        return ani  # Return animation object to prevent garbage collection
