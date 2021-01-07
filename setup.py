import numpy as np
import math
import os
import torch as T


from environment.simulator import Simulator
from ai.policy_gradient.Agent import Agent
from ai.policy_gradient.Reinforce import Reinforce



#Path to existing policy
policy_path = "policy.pth"
policy_device = "cuda:0"

#Define parameters
randomize_map = True
time_step = 16 #Set to None, to use frame time
map_size = np.array([1600, 900])


#Create environment
def create_simulator():
    sim = Simulator(map_size=map_size, 
                    map_front_segments=4, 
                    map_back_segments=2,
                    checkpoint_reward=2000, 
                    step_reward=-1,
                    lose_reward=-10000,
                    win_reward=20000, 
                    lap_amount=2, 
                    step_amount=12000,
                    agent_size=np.array([20, 10]), 
                    agent_sensor_angles=[0, math.pi/3, -math.pi/3, math.pi/9, -math.pi/9],
                    agent_sensor_lengths=[180, 180, 180, 180, 180])

    state = sim.get_state()

    return sim, state


#Create agent
def create_agent():
    #If pre-trained policy exists, load it
    if os.path.isfile(policy_path):
        policy = T.load(policy_path)
    #Else create new policy
    else:
        policy = Reinforce([5, 96, 64, 3], policy_device)

    #Return agent with policy
    return Agent(policy, learning_rate=6e-4,
                 future_discount=0.997,
                 replay_buffer_size=32000, replay_batch_size=9000)