import numpy as np
import math
import os
import torch as T


from environment.simulator import Simulator
from ai.policy_gradient.Agent import Agent
from ai.policy_gradient.Reinforce import Reinforce



#Policy related options
policy_path = "policy-snapshots/R{}-policy.pth"
freeze_snapshot_folder = "policy-snapshots/D{}-E{}/"
freeze_snapshot_file = "R{}-P{}-policy.pth"
epochs_per_freeze_snapshot = 1

epoch_training_iterations = 4
epoch_training_data = 6
epoch_training_replay = 3
epoch_offspring_per_elite = 3
epoch_elite = 5

#Plot related options
plot_data_path = "plot-data/current-plot.csv"
plot_x_axis = "Episodes simulated"
plot_y_axis = "Avg. summed batch reward"

#Policy training device
policy_device = "cpu"


#Define parameters
randomize_map = True
time_step = 16 #Set to None, to use frame time
map_size = np.array([1600, 900])


#Create environment
def create_simulator():
    sim = Simulator(map_size=map_size, 
                    map_front_segments=4, 
                    map_back_segments=1,
                    checkpoint_reward=2000, 
                    lose_reward=-10000,
                    speed_reward=6,
                    simulation_max_time=28800, 
                    checkpoint_max_time=1800,
                    agent_size=np.array([10, 5]), 
                    agent_sensor_angles=[0, math.pi/22, -math.pi/22, math.pi/5, -math.pi/5, math.pi*6/14, -math.pi*6/14],
                    agent_sensor_lengths=[220, 200, 200, 160, 160, 100, 100])

    state = sim.get_state()

    return sim, state


def create_policy(id):
    path = policy_path.format(id)

    #If pre-trained policy exists, load it
    if os.path.isfile(path):
        return T.load(path)
    #Else create new policy
    else:
        return Reinforce([9, 32, 32, 7], policy_device)

def create_policies():
    #Attempt to load or create each elite policy
    return [create_policy(i) for i in range(epoch_elite)]

#Create agent
def create_agent(policy):
    return Agent(policy, learning_rate=6e-4,
                 future_discount=0.997,
                 games_avg_store=epoch_training_data, games_avg_replay=epoch_training_replay,
                 replay_buffer_size=256000, replay_batch_size=9000)
