# Lunar Lander: AI-controlled play


from LunarLander import *

env = LunarLander()
env.reset()
exit_program = False

# Creating a NN : 
# Setting things up : 
import torch
import torch.nn
import random as r
import numpy as np
N = 64
input_neurons = 5
hidden_neurons = 6
output_neurons = 8

#Distance, speed, fuel, win
reward_factors = np.array([-1.0, -0.6, 0.1, 100])


# Creating network : 
model = torch.nn.Sequential(
            torch.nn.Linear(input_neurons, hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_neurons, output_neurons),
            torch.nn.Softmax()
        )

# Saving states 
states = []
actions = []
rewards = []


def action_to_index(boost, left, right):
    return 4*boost + 2*left + right

def index_to_action(index):
    return (index // 4, (index % 4) // 2, index % 2)

while not exit_program:
    env.render()

    #Get state vector
    input = torch.tensor([env.rocket.x, env.rocket.y, 
                          env.rocket.xspeed, env.rocket.yspeed, 
                          env.rocket.fuel])
    
    #Forward-pass through model
    action_probabilities = model(input).detach().numpy()

    #Randomly sample action from distribution
    action_index = np.random.choice(np.arange(output_neurons), p = action_probabilities)
    (boost, left, right) = index_to_action(action_index)

    #Step
    (x, y, x_speed, y_speed), reward, done = env.step((boost, left, right))
    fuel = env.rocket.fuel
    
    #Save state, action, reward
    states.append([x, y, x_speed, y_speed, fuel])
    actions.append([action_to_index(*(boost, left, right))])

    if done:
        won = env.won
        pos_delta = ((x - 400)**2 + (y-575)**2)**0.5
        speed = (x_speed**2 + y_speed**2)**0.5

        reward = reward_factors * np.array([pos_delta, speed, fuel, won]).sum()

        rewards.append(reward)
        
  
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True


    # left = r.choice([True, False])
    # right = r.choice([True, False])
    # boost = r.choice([True, False])
    
    # # Action from neural net : 
    # model_input = torch.Tensor((x, y, xspeed, yspeed))
    
    # model_output = model(model_input)
    
    # if model_output[0] - random.random() < 0:
    #     boost = True
    # else : 
    #     boost = False
    # if model_output[1] - random.random() < 0:
    #     left = True
    # else : 
    #     left = False
    # if model_output[2] - random.random() < 0:
    #     right = True
    # else :
    #     right = False      


env.close()