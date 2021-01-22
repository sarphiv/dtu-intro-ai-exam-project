from plotting.CsvManager import CsvManager
import pygame as pg
import numpy as np
import random as r
import pandas as pd
import math
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch as T

from setup import create_policies, create_policy, create_agent, action_interval, reward_factors

from environment.LunarLander import LunarLander
from game.action_space import index_to_action

import pickle

# Getting test starting positions : 
test_positions = pd.read_csv("starting_pos/random.csv").to_numpy()


def run_simulations(agent):
    # Running the test positions 3 times per position 
    scores = []
    
    for i in test_positions: 
        for _ in range(3): 
            j = 0
            env = LunarLander()
            (env.rocket.x, env.rocket.y, env.rocket.xspeed, env.rocket.yspeed, env.rocket.fuel) = tuple(i) 
            state = np.array(env.get_state())
            done = False
            
            while not done:
                #Get next action
                take_action = j % action_interval == 0
                if take_action:
                    action_id = agent.action(state)
                    action = index_to_action(action_id)

                #Simulate time step
                _, reward, done = env.step(action)
                state = np.array(env.get_state())
                
                j += 1
            
                if done:
                    (x, y, xspeed, yspeed, fuel) = state
                    pos_delta = max(abs(x) - 20, 0) # within winning range 
                    speed = (xspeed**2 + yspeed**2)**0.5 if not (xspeed**2 + yspeed**2)**0.5 - 20 < 0 else 0
                    fuel = env.rocket.fuel
                    won = env.won

                    reward = np.array([pos_delta**0.25, speed, fuel*won, won])
                    reward = (reward_factors * reward).sum()
                    
                    scores.append(reward)
              
    scores = np.array(scores)     
    # Returning the mean of the scores and also how many runs were "good-enough"
    return scores.mean(), len(scores[scores >= 4500])/90, len(scores[scores >= 2000])/90
            



for k in range(1, 31): 
    # Saving data : 
    plot_file_reward = CsvManager(["Games Played", "Avg Reward"], file_name=f"plot-data/plot_file_reward{k}.csv", clear=True)
    plot_file_won = CsvManager(["Games Played", "Procent Won"], file_name=f"plot-data/plot_file_won{k}.csv", clear=True)
    plot_file_won_game = CsvManager(["Games Played", "Procent Won"], file_name=f"plot-data/plot_file_won_game{k}.csv", clear=True)

    
    # Path for the policies : 
    path = f"./results/genetic-lunar{k}/policy-snapshots"

    # Get all policies (should be 100 long)
    directories = os.listdir(path)


    for i, c in enumerate(directories):
        # Getting the path for each policy saved 
        policy_path = path + "/" + c + "/" + os.listdir(path+"/"+c)[0]
        
        # temporary path to one of my policies 
        # policy = T.load("testing_policy/policy1.pth")
        # TODO comment ^ this out and v that in 
        
        # loading in the policy and creating agent 
        policy = T.load(policy_path)
        agent = create_agent(policy)
        
        # run the test simulations to see how good the policy is 
        score = list(run_simulations(agent))
        print(score)
        
        # There are 2*6*5*16 *4 = 38400 games between each policy in the genetic environment
        games_played = (i+1)*38400
        
        # Save the policy's score in the plot-data folder 
        plot_file_reward.save_data([games_played, score[0]])
        plot_file_won.save_data([games_played, score[1]])
        plot_file_won_game.save_data([games_played, score[2]])
    