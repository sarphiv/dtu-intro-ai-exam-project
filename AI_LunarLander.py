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

# Type run : 
learning_rate = 1e-3
expert_system = False # The AI is pretraind with an AI (for expert_train_time)
expert_train_time = 100000 # the expert system will be on the first 500000 games

is_soft_max = True # If True the last layer has a Softmax activation function, else a linear destribution 

paths = ["expertsystem/", "softmax/", "normelized/"] # path names for different folders 
run_id = "_1_e" + str(abs(int(np.log10(learning_rate))))  # increment each run 



#Game settings
game_train_freq = 50 # games before each training
action_freq = 4 # ticks before next action 
last_counter = 60 # Counter for running mean 


update_graph = True # Toggle to update the graf of running mean 

# Best rolling avg : 
best_rolling_avg = -1000000


#Path to existing policy and csv file 
if expert_system:
    policy_path = paths[0] + "policy" + run_id + ".pth" 
    best_policy_path = paths[0] + "best_policy" + run_id + ".pth"
    csv_file = csv_manager(["Games Played", "Avg Reward"], file_name=paths[0] + "data" + run_id + ".csv", clear=True)
elif is_soft_max:
    policy_path = paths[1] + "policy" + run_id + ".pth" 
    best_policy_path = paths[1] + "best_policy" + run_id + ".pth"
    csv_file = csv_manager(["Games Played", "Avg Reward"], file_name=paths[1] + "data" + run_id + ".csv", clear=True)
else: 
    policy_path = paths[2] + "policy" + run_id + ".pth" 
    best_policy_path = paths[2] + "best_policy" + run_id + ".pth"
    csv_file = csv_manager(["Games Played", "Avg Reward"], file_name=paths[2] + "data" + run_id + ".csv", clear=True)
    
plots = plot_manager([csv_file]) # created plot manager 
    
#If pre-trained policy exists, load it
if os.path.isfile(policy_path):
    policy = torch.load(policy_path)
#Else create new policy
else:
    policy = PolicyGradient([5, 64, 64, 6], "cpu", is_soft_max=is_soft_max)
    

#Create agent with policy
agent = Agent(policy, learning_rate=learning_rate,
              future_discount=0.9997, games_avg_store=80, games_avg_replay=36,
              replay_buffer_size=320000, replay_batch_size=12000)

#Distance, speed, fuel, win - reward factors 
reward_factors = np.array([-100, -60, 50, 2000]) 

#Initialize environment
env = LunarLander()
env.reset()
exit_program = False
render_game = True
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
validation_games_amount = 10
validation_interval = 2
is_validating = False # wether the environment is training or validating what it has learned

# Set validation case : 
def validation_case():
    # fuel = 100 
    # x = 300*(np.random.rand()*2-1)
    # y  = 400 
    # x_speed = 3*(np.random.rand()*2-1)
    # y_speed = 3*(np.random.rand()*2-1)
    
    # Certian values that we use across all trainings to compare 
    x = -276.187248078422
    y = 400 
    x_speed = 2.9723966095606857
    y_speed = 0.22020395570370832
    fuel = 100
    
    return (x,y,x_speed,y_speed,fuel)
    

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

# Expert system : 
def expert_action(x,y,x_speed,y_speed,fuel):
    boost = (x < 175 and y_speed>3 or x < 7 and y_speed > 2 or x < 1 and y_speed >= 0 )
    if abs(x)<=20 and y_speed < 19:
        boost = False

    if x < 0:
        left = True
        right = False
        if x < -70 and x_speed >= 20:
            right = False
            left = False
        if x >= -70 and x_speed >= 4:
            right = False
            left = False
        if x > -3:
            right = False
            left = False

    if 0 < x:
        right = True
        left = False
        if x > 70 and x_speed <= -20:
            right = False
            left = False
        if x < 70 and x_speed <= -4:
            right = False
            left = False
        if x < 3:
            right = False
            left = False
    
    return (boost, left, right)

#Game counters
game_counter = 0
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
        if not (expert_system and expert_train_time >= game_counter) : 
            action_id = agent.action(current_state)
            (boost, left, right) = index_to_action(action_id)
        else: 
            # expert system action
            (boost, left, right) = expert_action(current_state)
            action_id = action_to_index(boost, left, right)
        
        #Save state, action
        #NOTE: Reward is saved after taking action
        if not is_validating: 
            states.append(current_state)
            actions.append(action_id)

    #Step environment
    (x, y, x_speed, y_speed), _, done = env.step((boost, left, right))
    
    # Always append reward on each action (no intermediate rewards)
    if not done:
        if action_counter % action_freq == 0 and not is_validating:
            rewards.append(0)
    #Else game is done, store rewards
    else:
        #Calculate game rewards
        #pos_delta = abs(x - platform_center_x) # exactly on the center
        #speed = (x_speed**2 + y_speed**2)**0.5 
        pos_delta = max(abs(x - platform_center_x) - 20, 0) # within winning range 
        speed = (x_speed**2 + y_speed**2)**0.5 if not (x_speed**2 + y_speed**2)**0.5 - 20 < 0 else 0
        fuel = env.rocket.fuel
        won = env.won

        
        reward = np.array([pos_delta**0.25, speed, fuel*won, won])
        
        reward = (reward_factors * reward).sum()


        if not is_validating : 
            game_counter += 1
                   
            #Store rewards
            rewards.append(reward)

            #Save episode states, actions, and rewards
            agent.save_episode(states, actions, rewards)
            
            # List of the last games terminal rewards (could do sum(rewards) instead of reward down here)
            if len(last_rewards) <= last_counter: 
                last_rewards.append(reward)
            else: 
                last_rewards = last_rewards[1:] + [reward]
                
        #If training time, train and checkpoint on network
        if game_counter % game_train_freq == 0 and not is_validating:       
            # save games_played and the running mean  
            # avg_reward_last_games = sum(last_rewards)/len(last_rewards)    
            # #print(f"{game_counter}: {sum(last_rewards)/len(last_rewards)}")
            # csv_file.save_data([game_counter, avg_reward_last_games])
            # # Always store the best policy for each run (the one that did the best)
            # if avg_reward_last_games > best_rolling_avg: 
            #     torch.save(policy, best_policy_path)
            # # Update the graph 
            # if update_graph: 
            #     plots.plot()
            
            agent.train()
            torch.save(policy, policy_path)
            
            # Validate after every 'validation_interval' training
            if game_counter % (game_train_freq * validation_interval) == 0 : 
                is_validating = True

        elif is_validating: 
            validation_games.apppend(sum(reward))
            
            if len(validation_games) >= validation_games_amount: 
                is_validating = False
                #save games_played and the running mean  
                avg_reward = sum(validation_games)/len(validation_games)    
                #print(f"{game_counter}: {sum(last_rewards)/len(last_rewards)}")
                csv_file.save_data([game_counter, avg_reward])
                # Always store the best policy for each run (the one that did the best)
                if avg_reward > best_rolling_avg: 
                    torch.save(policy, best_policy_path)
                    best_rolling_avg = avg_reward
                # Update the graph 
                if update_graph: 
                    plots.plot()   
                
                # reset validation_games: 
                validation_games = []
                
                     
        #Clear episode memory
        states.clear()
        actions.clear()
        rewards.clear()

        #Reset game environment
        env.reset()
        
        if is_validating: 
            # setting the validation case : 
            (env.rocket.x, env.rocket.y, env.rocket.xspeed, env.rocket.yspeed, env.rocket.fuel) = validation_case()
            

    
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

env.close()