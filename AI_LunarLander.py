# Lunar Lander: AI-controlled play

import os
from Agent import Agent
from PolicyGradient import PolicyGradient
from ai_train import normalize_state, train
from LunarLander import *


env = LunarLander()
env.reset()
exit_program = False
render_game = True

# Creating a NN : 
# Setting things up : 
import torch
import torch.nn
import random as r
import numpy as np


platform_center_x = 0

action_counter = 0
action_freq = 4

policy_path = "policy.pth"

#Distance, speed, fuel, win
reward_factors = np.array([-100, -60, 10, 2000])

#If pre-trained policy exists, load it
if os.path.isfile(policy_path):
    policy = torch.load(policy_path)
#Else create new policy
else:
    policy = PolicyGradient([5, 64, 64, 6], 1e-3, "cuda:0")
    
agent = Agent(policy, future_discount=0.9997,
              replay_buffer_size=6000, replay_batch_size=3000)


# Saving states 
#[ [x, y, x_s, y_s, f], ... ]: List[List[float, float, float, float, float]]
states = []
#[ a, ... ]: List[int]
actions = []
#[ r, ... ]: List[float]
rewards = []

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

counter = 0
game_counter = 0
while not exit_program:
    counter += 1
    
    if render_game:
        env.render()
        
    action_counter += 1

    #Get state
    if action_counter % action_freq == 0:
        current_state = normalize_state(env.rocket.x, env.rocket.y, 
                                        env.rocket.xspeed, env.rocket.yspeed, 
                                        env.rocket.fuel)
        
        action_id = agent.action(current_state)
        (boost, left, right) = index_to_action(action_id)
        
        #Save state, action
        #NOTE: Reward is saved after taking action
        states.append(current_state)
        actions.append(action_id)

    #Step
    (x, y, x_speed, y_speed), _, done = env.step((boost, left, right))
    fuel = env.rocket.fuel
    
    # Always append reward (no intermediate rewards)
    if not done:
        if action_counter % action_freq == 0:
            rewards.append(0)
    else:
        game_counter += 1
        
        # If the game is over we store the terminal rewards
        won = env.won
        pos_delta = abs(x - platform_center_x)
        speed = (x_speed**2 + y_speed**2)**0.5

        reward = (reward_factors * np.array([pos_delta, speed, fuel, won])).sum()
                
        rewards.append(reward)

        agent.save_episode(states, actions, rewards)
        

        # When done train the network and clear lists  
        if game_counter % 10 == 0:
            print(f"{game_counter}: {reward}")
            agent.train()
            torch.save(policy, policy_path)


        states.clear()
        actions.clear()
        rewards.clear()

        env.reset()
        # env.rocket.x = platform_center_x
        # env.rocket.y = 300
        # env.rocket.xspeed = 0
        # env.rocket.yspeed = 0

    
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                render_game = not render_game

env.close()