import numpy as np
import math
import os
import torch as T


from environment.simulator import Simulator
from ai.policy_gradient.Agent import Agent
from ai.policy_gradient.Reinforce import Reinforce



#Path to policies
policy_path = "policy-snapshots/current-policy.pth"
freeze_snapshot_path = "policy-snapshots/D{}-E{}-P{}-policy.pth"

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


#Create agent
def create_agent():
    #If pre-trained policy exists, load it
    if os.path.isfile(policy_path):
        policy = T.load(policy_path)
    #Else create new policy
    else:
        policy = Reinforce([9, 32, 32, 7], policy_device) #NOTE: Not allowing turning on the spot

    #Return agent with policy
    return Agent(policy, learning_rate=1e-3,
                 future_discount=0.997,
                 games_avg_store=2, games_avg_replay=1,
                 replay_buffer_size=1920000, replay_batch_size=9000)
