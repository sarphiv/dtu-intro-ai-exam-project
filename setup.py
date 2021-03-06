import numpy as np
import math
import os
import torch as T


from environment.simulator import Simulator
from ai.policy_gradient.Agent import Agent
from ai.policy_gradient.Reinforce import Reinforce


#Training state
state_path = "state.txt"

#Policy related options
policy_folder = "policy-snapshots/"
policy_path = policy_folder + "R{}-policy.pth"
freeze_snapshot_folder = policy_folder + "D{}-E{}/"
freeze_snapshot_file = "R{}-P{}-policy.pth"
epochs_per_freeze_snapshot = 1

epoch_training_iterations = 3
epoch_training_data = 4
epoch_training_replay = 3
epoch_evaluation = 3
epoch_offspring_per_elite = 4
epoch_elite = 12

max_parallelism = 28
max_generations = 150

simulation_max_time = 28800

#Plot related options
plot_data_folder = "plot-data/"
plot_data_path = plot_data_folder + "current-plot.csv"
plot_x_axis = "Generations simulated"
plot_y_axis = "Best elite mean reward"

#Policy training device
policy_device = "cpu"


#Define parameters
randomize_map = True
#NOTE: Physics is tuned for 16 ms. Higher time step to generalize training.
time_step = 16 #Set to None, to use frame time.
map_size = np.array([1600, 900])


#Create environment
def create_simulator(map_id=None, map_direction=None):
    sim = Simulator(map_size=map_size, 
                    map_front_segments=4, 
                    map_back_segments=1,
                    checkpoint_reward=2000, 
                    lose_reward=-10000,
                    speed_reward=6,
                    simulation_max_time=simulation_max_time, 
                    checkpoint_max_time=1800,
                    agent_size=np.array([10, 5]), 
                    agent_sensor_angles=[0, math.pi/20, -math.pi/20, math.pi/8, -math.pi/8, math.pi/5, -math.pi/5, math.pi*4/14, -math.pi*4/14],
                    agent_sensor_lengths=[220, 180, 180, 180, 180, 160, 160, 140, 140],
                    map_id=map_id,
                    map_direction=map_direction)

    state = sim.get_state()

    return sim, state


def create_policy(id):
    path = policy_path.format(id)

    #If pre-trained policy exists, load it
    if os.path.isfile(path):
        return T.load(path)
    #Else create new policy
    else:
        return Reinforce([10, 128, 64, 7], policy_device)

def create_policies():
    #Attempt to load or create each elite policy
    return [create_policy(i) for i in range(epoch_elite)]

#Create agent
def create_agent(policy):
    return Agent(policy, learning_rate=6e-4,
                 future_discount=0.997,
                 games_avg_store=epoch_training_data, games_avg_replay=epoch_training_replay,
                 replay_buffer_size=simulation_max_time*epoch_training_data, replay_batch_size=9000)
