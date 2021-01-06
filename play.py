import pygame as pg
import numpy as np
import random as r
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch as T

from environment.simulator import Simulator
from game.keyboard_controller import create_keyboard_controller, wasd_control_scheme, uhjk_control_scheme
from game.drawers import draw_game
from ai.policy_gradient.Agent import Agent
from ai.policy_gradient.PolicyGradient import PolicyGradient



#Define parameters
running = True
render = True
randomize_map = True
time_step = 7 #Set to None, to use frame time
game_training_interval = 10
map_size = np.array([1600, 900])

pg.init()
window = pg.display.set_mode(tuple(map_size))
clock = pg.time.Clock()

game_counter = 0
time_delta = time_step
pg_events = None


#Helper function to tidy up things
def handle_events():
    global render
    global randomize_map
    global time_delta
    global pg_events
    global running
    
    #Get time and events
    time_delta = time_step if time_step else clock.tick()
    pg_events = pg.event.get()

    #Handle events
    for e in pg_events:
        if e.type == pg.QUIT:
            running = False
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_r:
                render = not render
            if e.key == pg.K_m:
                randomize_map = not randomize_map



#Create environment
sim = Simulator(map_size=map_size, 
                map_front_segments=4, 
                map_back_segments=2,
                checkpoint_reward=2000, 
                step_reward=-1,
                lose_reward=-10000,
                win_reward=20000, 
                lap_amount=3, 
                step_amount=120000,
                agent_size=np.array([20, 10]), 
                agent_sensor_angles=[0, math.pi/3, -math.pi/3, math.pi/9, -math.pi/9],
                agent_sensor_lengths=[180, 180, 180, 180, 180])

#Create temporary keyboard controller
# controller = create_keyboard_controller(lambda: pg_events, wasd_control_scheme)

#Path to existing policy
policy_path = "policy.pth"

#If pre-trained policy exists, load it
if os.path.isfile(policy_path):
    policy = T.load(policy_path)
#Else create new policy
else:
    policy = PolicyGradient([7, 32, 16, 3], "cuda:0")


#Create agent with policy
agent = Agent(policy, learning_rate=6e-4,
              future_discount=0.9997,
              replay_buffer_size=30000, replay_batch_size=3000)


# Saving states 
#[ [x, y, x_s, y_s, f], ... ]: List[List[float, float, float, float, float]]
states = []
#[ a, ... ]: List[int]
actions = []
#[ r, ... ]: List[float]
rewards = []

#Preinitialize first state
state = sim.get_state()

#Game loop
while running:
    #Handle pygame events
    handle_events()



    #Get next action
    action = agent.action(state)
    # action = controller()

    #Simulate time step
    state, reward, done = sim.step(time_delta, action + 6)
    # state, reward, done = sim.step(time_delta, action)
    
    states.append(state)
    actions.append(action)
    rewards.append(reward)


    #If agent won/lost, reset environment
    if done:
        #Increment game counter
        game_counter += 1
        
        print(f"{game_counter}: {np.array(rewards).sum()}")
        
        agent.save_episode(states, actions, rewards)
        
        states.clear()
        actions.clear()
        rewards.clear()
        
        if game_counter % game_training_interval == 0:
        
            agent.train()
            T.save(policy, policy_path)


        if randomize_map:
            sim.reset()
        else:
            sim.reset(sim.map_id, sim.map_direction)



    #If rendering enabled, draw game
    draw_game(render, window, sim, state)

