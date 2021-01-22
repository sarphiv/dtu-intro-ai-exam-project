# Lunar Lander: AI-controlled play

import os
import torch
import torch.nn
import numpy as np
from Agent import Agent
from PolicyGradient import PolicyGradient
from LunarLander import *
from csv_manager import csv_manager
from plot_manager import plot_manager
import pandas as pd


# Load disired agent 
policy_path =  "data/policy1.pth"
    
# #If pre-trained policy exists, load it
if os.path.isfile(policy_path):
    policy = torch.load(policy_path)
#Else create new policy
else:
    policy = PolicyGradient([5, 64, 64, 6], "cpu", is_soft_max=True)

#Create agent with policy
agent = Agent(policy, learning_rate=1e-4,
            future_discount=0.9997, games_avg_store=80, games_avg_replay=36,
            replay_buffer_size=320000, replay_batch_size=12000)
    
    
action_freq = 4 # ticks before next action 
render_game = True

#Initialize environment
env = LunarLander()
env.reset()
exit_program = False
    
#Helper methods
def action_to_index(boost, left, right):
    #TODO: Fix this mess, I'm tired of things not working,
    # SO THINGS JUST NEED TO WORK NOW
    return {
        (False, False, False): 0,
        (False, True, True): 0,
        (True, False, False): 1,
        (True, True, True): 1,
        (True, True, False): 2,
        (True, False, True): 3,
        (False, True, False): 4,
        (False, False, True): 5,
    }[boost, left, right]

def index_to_action(index):
    #TODO: Fix this mess, I'm tired of things not working,
    # SO THINGS JUST NEED TO WORK NOW
    return {
        0: (False, False, False),
        1: (True, False, False),
        2: (True, True, False),
        3: (True, False, True),
        4: (False, True, False),
        5: (False, False, True),
    }[index]
    
#Game counters
action_counter = 0
while not exit_program:
    
    if render_game:
        env.render()
        
    action_counter += 1

    #If agent should act, get next action and save it
    if action_counter % action_freq == 0:
        #Collect current state
        current_state = [env.rocket.x, env.rocket.y, 
                         env.rocket.xspeed, env.rocket.yspeed, 
                         env.rocket.fuel]
        
        #Get action based on current state 
        action_id = agent.action(current_state)
        (boost, left, right) = index_to_action(action_id)
        

    #Step environment
    (x, y, x_speed, y_speed), _, done = env.step((boost, left, right))
    
    if done:
        #Reset game environment
        env.reset()
            

    
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                render_game = not render_game
            if event.key == pygame.K_e:
                expert_system = not expert_system
            if event.key == pygame.K_g:
                update_graph = not update_graph
            if event.key == pygame.K_h:
                view += 1
            if event.key == pygame.K_v:
                is_validating = True
                validation_counter = 0

env.close()