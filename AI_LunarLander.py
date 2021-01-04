# Lunar Lander: AI-controlled play

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

action_rate = 4

input_neurons = 5
hidden_neurons1 = 64
hidden_neurons2 = 64
#hidden_neurons3 = 64
output_neurons = 8

platform_center_x = 400

#Distance, speed, fuel, win
#reward_factors = np.array([-2.0, -0.2, 0.1, 100])
reward_factors = np.array([-100, 0, 0, 100])


# Creating network :
model = torch.nn.Sequential(
            torch.nn.Linear(input_neurons, hidden_neurons1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_neurons1, hidden_neurons2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_neurons2, output_neurons),
            torch.nn.Softmax(),

            #torch.nn.Sigmoid(),
        )

# Saving states 
#[ [x, y, x_s, y_s, f], ... ]: List[List[float, float, float, float, float]]
states = []
#[ a, ... ]: List[int]
actions = []
#[ r, ... ]: List[float]
rewards = []
#[ t, ... ]: List[int] (list of terminal state indexes)
terminals = [ ]


def action_to_index(boost, left, right):
    return 4*boost + 2*left + right

def index_to_action(index):
    return (index // 4, (index % 4) // 2, index % 2)

counter = 0
game_counter = 0
while not exit_program:
    counter += 1
    
    if render_game:
        env.render()
        
        
    #TODO: Can we normalize inputs?
    if counter % action_rate == 0:
        #Get state
        current_state = normalize_state(env.rocket.x, env.rocket.y, 
                                        env.rocket.xspeed, env.rocket.yspeed, 
                                        env.rocket.fuel)
        input = torch.tensor(current_state)
    
        #Forward-pass through model
        # action_probabilities = model(input).detach().numpy()
        action_probabilities = model(input).detach().numpy()
        # action_probabilities = action_probabilities / action_probabilities.sum()

        #Randomly sample action from distribution
        action_index = np.random.choice(np.arange(output_neurons), p = action_probabilities)
        (boost, left, right) = index_to_action(action_index)

    #Step
    (x, y, x_speed, y_speed), reward, done = env.step((boost, left, right))
    fuel = env.rocket.fuel
        
        
    if counter % action_rate == 0:
        #Save state, action, reward
        states.append(current_state)
        actions.append([action_to_index(*(boost, left, right))])
        
        # Always append reward (no non-terminal rewards)
        reward = 0
        
        # If the game is over we store the terminal rewards 
        if done:
            game_counter += 1
            won = env.won
            pos_delta = abs(x - platform_center_x)
            speed = (x_speed**2 + y_speed**2)**0.5

            reward = (reward_factors * np.array([pos_delta, speed, fuel, won])).sum()
            # Adds the index of the game 
            terminals.append(len(rewards))
            
        rewards.append(reward)
        
        # When done train the network and clear lists  
        if done:
            if game_counter % 50 == 0:
                train(model, states, actions, rewards, terminals)
                states.clear()
                actions.clear()
                rewards.clear()
                terminals.clear()
        
            env.reset()
            env.rocket.x = 0
            env.rocket.y = 300
            env.rocket.xspeed = 0
            env.rocket.yspeed = 0

    
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                render_game = not render_game

env.close()