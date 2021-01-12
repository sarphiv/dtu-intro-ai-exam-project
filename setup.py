import numpy as np
import math
import os
import torch as T


from ai.policy_gradient.Agent import Agent
from ai.policy_gradient.Reinforce import Reinforce



#Policy related options
policy_path = "policy-snapshots/R{}-policy.pth"
freeze_snapshot_folder = "policy-snapshots/D{}-E{}/"
freeze_snapshot_file = "R{}-P{}-policy.pth"
epochs_per_freeze_snapshot = 4

epoch_training_iterations = 2
epoch_training_data = 32
epoch_training_replay = 32
epoch_evaluation = 2
epoch_offspring_per_elite = 2
epoch_elite = 6

max_generations = 100

simulation_max_time = 28800

#Plot related options
plot_data_path = "plot-data/current-plot.csv"
plot_x_axis = "Episodes simulated"
plot_y_axis = "Avg. summed batch reward"

#Policy training device
policy_device = "cpu"


#Define parameters
randomize_map = True
reward_factors = np.array([-100, -60, 50, 2000]) 
action_interval = 4



def create_policy(id):
    path = policy_path.format(id)

    #If pre-trained policy exists, load it
    if os.path.isfile(path):
        return T.load(path)
    #Else create new policy
    else:
        return Reinforce([5, 64, 64, 6], policy_device)

def create_policies():
    #Attempt to load or create each elite policy
    return [create_policy(i) for i in range(epoch_elite)]

#Create agent
def create_agent(policy):
    return Agent(policy, learning_rate=6e-4,
                 future_discount=0.997,
                 games_avg_store=epoch_training_data, games_avg_replay=epoch_training_replay,
                 replay_buffer_size=simulation_max_time*epoch_training_data, replay_batch_size=9000)
