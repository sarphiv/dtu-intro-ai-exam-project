import numpy as np
import math
import os
import torch as T


from environment.simulator import Simulator
from ai.policy_gradient.Agent import Agent
from ai.policy_gradient.Reinforce import Reinforce



#Path to existing policy
policy_path = "policy-snapshots/current-policy.pth"
freeze_snapshot_path = "policy-snapshots/T{}-P{}-policy.pth"
policy_device = "cuda:0"

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
                    time_reward=-3,
                    lose_reward=-10000,
                    win_reward=20000, 
                    lap_amount=1, 
                    checkpoint_max_time=1200,
                    agent_size=np.array([10, 5]), 
                    agent_sensor_angles=[0, math.pi/24, -math.pi/24, math.pi/4, -math.pi/4, math.pi*6/14, -math.pi*6/14],
                    agent_sensor_lengths=[220, 200, 200, 140, 140, 100, 100])

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
    return Agent(policy, learning_rate=6e-4,
                 future_discount=0.997,
                 games_avg_store=16, games_avg_replay=16,
                 replay_buffer_size=640000, replay_batch_size=9000)
