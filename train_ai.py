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


def start_new_training(run=0):
    # Type run : 
    learning_rate = 1e-4
    
    paths =  "data/" 
    #run_id = "_" + str(run) + "_e" + str(abs(int(np.log10(learning_rate))))  # increment each run 
    run_id = str(run)
    
    #Path to existing policy and csv file
    policy_path = paths + "policy" + run_id + ".pth" 
    best_policy_path = paths + "best_policy" + run_id + ".pth"

    # Save different data : 
    csv_random_reward = csv_manager(["Games Played", "Avg Reward"], file_name=paths + "random_reward_" + run_id + ".csv", clear=True)
    csv_random_won = csv_manager(["Games Played", "Procentage Won"], file_name=paths + "random_won_" + run_id + ".csv", clear=True)

    csv_choosen = csv_manager(["Games Played", "Avg Reward"], file_name=paths + "choosen_" + run_id + ".csv", clear=True)
    csv_outside = csv_manager(["Games Played", "Avg Reward"], file_name=paths + "outside_" + run_id + ".csv", clear=True)
    
    # #If pre-trained policy exists, load it
    # if os.path.isfile(policy_path):
    #     policy = torch.load(policy_path)
    # #Else create new policy
    # else:
    #     policy = PolicyGradient([5, 64, 64, 6], "cpu", is_soft_max=True)
    policy = PolicyGradient([5, 64, 64, 6], "cpu", is_soft_max=True)
    
    #Create agent with policy
    agent = Agent(policy, learning_rate=learning_rate,
                future_discount=0.9997, games_avg_store=80, games_avg_replay=36,
                replay_buffer_size=320000, replay_batch_size=12000)
    
    return (agent, policy, policy_path, best_policy_path, csv_random_reward, csv_random_won, csv_choosen, csv_outside)
    

#Game settings
game_train_freq = 50 # games before each training
action_freq = 4 # ticks before next action 
last_counter = 60 # Counter for running mean 

update_graph = True # Toggle to update the graf of running mean 
view = 0 
render_game = True

# Best rolling avg : 
best_rolling_avg = -1000000
good_enough = 4500

#Distance, speed, fuel, win - reward factors 
reward_factors = np.array([-100, -60, 50, 2000]) 

#Initialize environment
env = LunarLander()
env.reset()
exit_program = False
platform_center_x = 0

# Saving states 
#[ [x, y, x_s, y_s, f], ... ]: List[List[float, float, float, float, float]]
states = []
#[ a, ... ]: List[int]
actions = []
#[ r, ... ]: List[float]
rewards = []
last_rewards = []

# List of the rewards during the validation runs 
validation_games = []
validation_interval = 20
is_validating = False # wether the environment is training or validating what it has learned
validation_sets_names = ["random.csv", "choosen.csv", "outside.csv"]
validation_sets = []

for i in validation_sets_names: 
    validation_sets.append(pd.read_csv("starting_pos/" + i).to_numpy())
    
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
training_complete = True 
training_counter = 0
while not exit_program:
    if training_complete:
        game_counter = 0 
        (agent, policy, policy_path, best_policy_path, csv_random_reward, csv_random_won, csv_choosen, csv_outside) = start_new_training(training_counter)
        training_counter += 1
        training_complete = False
    
    # if render_game:
    #     env.render()
        
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
        
        #Save state, action
        #NOTE: Reward is saved after taking action
        if not is_validating: 
            states.append(current_state)
            actions.append(action_id)

    #Step environment
    (x, y, x_speed, y_speed), _, done = env.step((boost, left, right))
    
    # Always append reward on each action (no intermediate rewards)
    if not done:
        if action_counter % action_freq == 0:
            rewards.append(0)
    #Else game is done, store rewards
    else:
        pos_delta = max(abs(x - platform_center_x) - 20, 0) # within winning range 
        speed = (x_speed**2 + y_speed**2)**0.5 if not (x_speed**2 + y_speed**2)**0.5 - 20 < 0 else 0
        fuel = env.rocket.fuel
        won = env.won        
        reward = np.array([pos_delta**0.25, speed, fuel*won, won])
        reward = (reward_factors * reward).sum()

        #Store rewards
        rewards.append(reward)

        if not is_validating : 
            game_counter += 1

            #Save episode states, actions, and rewards
            agent.save_episode(states, actions, rewards)

        if game_counter % game_train_freq == 0 and not is_validating:                   
            agent.train()
            torch.save(policy, policy_path)
            
            # Validate after every 'validation_interval' training
            if game_counter % (game_train_freq * validation_interval) == 0 : 
                validation_counter = 0
                is_validating = True

        elif is_validating: 
            validation_games.append(sum(rewards))
            validation_counter += 1
            
            if len(validation_games) % 90 == 0: 
                #save games_played and the running mean  
                avg_reward = sum(validation_games)/len(validation_games)    
                #print(f"{game_counter}: {sum(last_rewards)/len(last_rewards)}")
                if validation_counter == 90: 
                    csv_random_reward.save_data([game_counter, avg_reward])
                    procent_won = 100*np.size(np.where(np.array(validation_games) > good_enough))/90
                    csv_random_won.save_data([game_counter, procent_won]) 
                    # Always store the best policy for each run (the one that did the best)
                    if avg_reward > best_rolling_avg: 
                        torch.save(policy, best_policy_path)
                        best_rolling_avg = avg_reward
                    
                    #printing 4 dtu : 
                    print(f"Games played : {game_counter},  avg reward : {avg_reward}", flush = True)
                        
                elif validation_counter == 180: 
                    csv_choosen.save_data([game_counter, avg_reward])
                elif validation_counter == 270: 
                    csv_outside.save_data([game_counter, avg_reward])
                    is_validating = False
                                
                    if procent_won >= 100:
                        exit_program = True
                        training_complete = True 
                
                    # # Update the graph 
                    # if update_graph: 
                    #     plot_random_reward.plot()   
                    #     plot_random_won.plot()   
                    #     plot_choosen.plot()   
                    #     plot_outside.plot()  
                # reset validation_games: 
                validation_games = []
                
                     
        #Clear episode memory
        states.clear()
        actions.clear()
        rewards.clear()

        #Reset game environment
        env.reset()
        
        if is_validating: 
            # set next training track 
            (env.rocket.x, env.rocket.y, env.rocket.xspeed, env.rocket.yspeed, env.rocket.fuel) = tuple(validation_sets[validation_counter//3//30][validation_counter//3 % 30])
            
            

    
    # # Process game events
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         exit_program = True
    #     if event.type == pygame.KEYDOWN:
    #         if event.key == pygame.K_r:
    #             render_game = not render_game
    #         if event.key == pygame.K_e:
    #             expert_system = not expert_system
    #         if event.key == pygame.K_g:
    #             update_graph = not update_graph
    #         if event.key == pygame.K_h:
    #             view += 1
    #         if event.key == pygame.K_v:
    #             is_validating = True
    #             validation_counter = 0

env.close()