import pygame as pg
import numpy as np
import random as r
import math
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch as T

from setup import create_policies, create_policy, create_agent, action_interval

from environment.LunarLander import LunarLander
from game.action_space import index_to_action


#Define parameters
running = True
restart = False





game_counter = 0


#Create simulation
env = LunarLander()
state = np.array(env.get_state())
#Load best agent
agent = create_agent(create_policy(0))

# controller = create_keyboard_controller(lambda: pg_events, wasd_control_scheme)
# controller = create_keyboard_controller(lambda: pg_events, dvorak_wasd_control_scheme)


#Helper functions to tidy up things
def handle_events():
    global randomize_map
    global running
    # global restart

    #Handle events
    for e in pg.event.get():
        if e.type == pg.QUIT:
            running = False
        if e.type == pg.KEYDOWN:
            pass
            # if e.key == pg.K_m:
            #     randomize_map = not randomize_map
            # if e.key == pg.K_r:
            #     restart 
                    

#Game loop

i = 0
while running:
    #Handle game events
    handle_events()
    
    env.render()
    

    #Get next action
    take_action = i % action_interval == 0
    if take_action:
        action_id = agent.action(state)
        action = index_to_action(action_id)

    #Simulate time step
    _, reward, done = env.step(action)
    state = np.array(env.get_state())
    
    i += 1


    #If agent won/lost, renew agent and reset rendered environment
    if done or restart:
        #Retrieve agent from file again in case of updates
        #NOTE: If failed to load agent from file, tries again.
        # Can happen if file is being updated.
        while True:
            try:
                agent = create_agent(create_policy(0))
                break
            except:
                print("Failed while loading agent")
                time.sleep(1.6)
                continue
        restart = False
        #Reset environment
        env.reset()
        state = np.array(env.get_state())
