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

#Initialize environment
env = LunarLander()
env.reset()
exit_program = False
render_game = True
expert_system = False
update_graph = True

#Game settings
action_freq = 4
game_train_freq = 50
platform_center_x = 0
last_counter = 60


#Path to existing policy
policy_path = "policy.pth"

#Distance, speed, fuel, win
reward_factors = np.array([-100, -60, 10, 2000])

#If pre-trained policy exists, load it
if os.path.isfile(policy_path):
    policy = torch.load(policy_path)
#Else create new policy
else:
    policy = PolicyGradient([5, 64, 64, 6], "cpu")


#Create agent with policy
agent = Agent(policy, learning_rate=1e-4,
              future_discount=0.9997,
              replay_buffer_size=9000, replay_batch_size=3000)

# Saving the scores and plotting tools 
csv_file1 = csv_manager(["Games Played", "Avg Reward"], file_name="testing1.csv", clear=False) # this contains 300000 games trained while already good 
csv_file2 = csv_manager(["Games Played", "Avg Reward"], file_name="testing2.csv", clear=False) # this is 350000 games converged but not solving -4000 score
csv_file3 = csv_manager(["Games Played", "Avg Reward"], file_name="testing3.csv", clear=False) # this is 500000 games that converged 3 times on the perfect engine
csv_file4 = csv_manager(["Games Played", "Avg Reward"], file_name="testing4.csv", clear=False) # Up's and downs
csv_file5 = csv_manager(["Games Played", "Avg Reward"], file_name="testing5.csv", clear=False) # 2.4 million games converted on the best after 1.5 mil? 
csv_file = csv_manager(["Games Played", "Avg Reward"], file_name="testing6.csv", clear=True)
# If we already have something to compare to : 
# csv_file2 = csv_manager(["Games Played", "Avg Reward"], file_name="NAME OF THAT FILE.csv", clear=False)
plots = plot_manager([csv_file1,csv_file2,csv_file3,csv_file4,csv_file5,csv_file]) # remeber adding other csv_managers if they need to be plotted 

# Saving states 
#[ [x, y, x_s, y_s, f], ... ]: List[List[float, float, float, float, float]]
states = []
#[ a, ... ]: List[int]
actions = []
#[ r, ... ]: List[float]
rewards = []
last_rewards = []

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
        if not expert_system: 
            action_id = agent.action(current_state)
            (boost, left, right) = index_to_action(action_id)
        else: 
            boost = ( env.rocket.x < 175 and env.rocket.yspeed>3 or env.rocket.x < 7 and env.rocket.yspeed > 2 or env.rocket.x < 1 and env.rocket.yspeed >= 0 )
            if abs(env.rocket.x)<=20 and env.rocket.yspeed < 19:

                boost = False

            if env.rocket.x < 0:
                left = True
                right = False
                if env.rocket.x < -70 and env.rocket.xspeed >= 20:
                    right = False
                    left = False
                if env.rocket.x >= -70 and env.rocket.xspeed >= 4:
                    right = False
                    left = False
                if env.rocket.x > -3:
                    right = False
                    left = False

            if 0 < env.rocket.x:
                right = True
                left = False
                if env.rocket.x > 70 and env.rocket.xspeed <= -20:
                    right = False
                    left = False
                if env.rocket.x < 70 and env.rocket.xspeed <= -4:
                    right = False
                    left = False
                if env.rocket.x < 3:
                    right = False
                    left = False
            action_id = action_to_index(boost, left, right)
        
        #Save state, action
        #NOTE: Reward is saved after taking action
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
        game_counter += 1
        
        #Calculate game rewards
        pos_delta = abs(x - platform_center_x)
        speed = (x_speed**2 + y_speed**2)**0.5
        fuel = env.rocket.fuel
        won = env.won
        
        reward = np.array([pos_delta**0.25, speed, fuel, won])
        #print(pos_delta**0.25*reward_factors[0], speed*reward_factors[1], fuel*reward_factors[2], won*reward_factors[3])
        #print(reward_factors * reward)
        reward = (reward_factors * reward).sum()

        #Store rewards
        rewards.append(reward)

        #Save episode states, actions, and rewards
        agent.save_episode(states, actions, rewards)
        
        if len(last_rewards) < last_counter: 
            last_rewards.append(reward)
        else: 
            last_rewards = last_rewards[1:] + [reward]
                
        #If training time, train and checkpoint on network
        if game_counter % game_train_freq == 0:       
            # save games_played and the running mean  
            avg_reward_last_games = sum(last_rewards)/len(last_rewards)    
            #print(f"{game_counter}: {sum(last_rewards)/len(last_rewards)}")
            csv_file.save_data([game_counter, avg_reward_last_games])
            
            # Update the graph 
            if update_graph: 
                plots.plot()
            
            agent.train()
            torch.save(policy, policy_path)

        
        #Clear episode memory
        states.clear()
        actions.clear()
        rewards.clear()

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

env.close()